"""Operational views: accounts, queue depth, cluster state, orphans, budget.

Everything here reads. The queue, the poller and the cluster connection belong
to other people's modules, and this panel observes them through interfaces they
already expose -- the RabbitMQ management API and the tasks table -- rather than
reaching into their code.
"""

import asyncio
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app import db, legacy_auth
from app.security import CurrentUser, require_admin
from app.services import outbox
from app.settings import settings

router = APIRouter(prefix="/api/admin", tags=["admin"])


class ApprovalRequest(BaseModel):
    user_id: str


@router.get("/users")
async def users(_: CurrentUser = Depends(require_admin)) -> dict:
    pending = await legacy_auth.list_unverified()
    everyone = await db.fetch_all(
        """
        SELECT id, email, role::text AS role, display_name,
               associated_organisation, join_reason, is_active,
               created_at, last_login_at,
               (SELECT COUNT(*) FROM tasks t WHERE t.submitted_by = users.id) AS runs
          FROM users ORDER BY created_at DESC LIMIT 500
        """
    )
    return {
        "pending": [
            {
                "id": str(u.id), "email": u.email, "display_name": u.display_name,
                "associated_organisation": u.associated_organisation,
                "join_reason": u.join_reason, "created_at": u.created_at,
            }
            for u in pending
        ],
        "users": everyone,
    }


@router.post("/users/approve")
async def approve(
    payload: ApprovalRequest, _: CurrentUser = Depends(require_admin)
) -> dict:
    try:
        account = await legacy_auth.approve_user(payload.user_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return {"id": str(account.id), "role": account.role}


@router.get("/budget")
async def budget(_: CurrentUser = Depends(require_admin)) -> dict:
    """Consumption per user.

    Reported in the currencies the evaluator actually counts. Wall time is shown
    but kept last and labelled, because §8 deprecates it as a comparison metric
    -- it measures which node the scheduler handed out, not the optimizer.
    """
    rows = await db.fetch_all(
        """
        SELECT u.id, u.email, u.display_name, u.role::text AS role,
               COUNT(t.task_id)                                  AS runs,
               COUNT(*) FILTER (WHERE t.task_status = 'failed')  AS failed,
               COUNT(*) FILTER (WHERE t.task_status IN ('pending','running')) AS active,
               COUNT(*) FILTER (WHERE t.created_at::date = CURRENT_DATE)      AS today,
               COALESCE(SUM(r.gradient_count), 0)                AS gradients,
               COALESCE(SUM(r.database_reaches), 0)              AS samples,
               COALESCE(SUM(r.wall_time_seconds), 0)             AS compute_seconds
          FROM users u
          LEFT JOIN tasks t   ON t.submitted_by = u.id
          LEFT JOIN results r ON r.task_id = t.task_id
         GROUP BY u.id, u.email, u.display_name, u.role
         HAVING COUNT(t.task_id) > 0
         ORDER BY COALESCE(SUM(r.database_reaches), 0) DESC
        """
    )
    return {"rows": rows, "daily_limit": settings.daily_submission_limit}


async def _rabbitmq() -> dict[str, Any]:
    if not settings.rabbitmq_management_url:
        return {"available": False, "reason": "RABBITMQ_MANAGEMENT_URL nie jest ustawiony"}
    try:
        async with httpx.AsyncClient(
            timeout=4.0,
            auth=(settings.rabbitmq_user, settings.rabbitmq_password),
        ) as client:
            response = await client.get(f"{settings.rabbitmq_management_url}/api/queues")
            response.raise_for_status()
            queues = response.json()
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    return {
        "available": True,
        "queues": [
            {
                "name": q.get("name"),
                "messages": q.get("messages", 0),
                "ready": q.get("messages_ready", 0),
                "unacknowledged": q.get("messages_unacknowledged", 0),
                "consumers": q.get("consumers", 0),
                "is_dlq": str(q.get("name", "")).upper().startswith("DLQ"),
            }
            for q in queues
        ],
    }


async def _orphans() -> list[dict]:
    """Runs whose observed state has stopped moving.

    §18 names the mechanism: a worker that dies after sbatch but before writing
    the job id leaves a SLURM job nobody is watching and which still burns
    grant. The symptom visible from here is a row that has been running or
    pending for far longer than a job of this size takes.
    """
    return await db.fetch_all(
        """
        SELECT task_id, run_name, task_status::text AS task_status,
               executor_task_id, submitted_by, created_at, updated_at,
               EXTRACT(EPOCH FROM (NOW() - updated_at)) AS stale_seconds
          FROM tasks
         WHERE task_status IN ('pending', 'running')
           AND updated_at < NOW() - INTERVAL '2 hours'
         ORDER BY updated_at ASC LIMIT 100
        """
    )


@router.get("/queue")
async def queue(_: CurrentUser = Depends(require_admin)) -> dict:
    broker, orphans, pending_outbox, recent_outbox = await asyncio.gather(
        _rabbitmq(), _orphans(), outbox.pending_count(), outbox.recent(25)
    )

    slurm = await db.fetch_all(
        """
        SELECT task_id, run_name, executor_task_id, task_status::text AS task_status,
               created_at, queued_at, started_at,
               EXTRACT(EPOCH FROM (NOW() - COALESCE(started_at, queued_at, created_at)))
                   AS elapsed_seconds
          FROM tasks
         WHERE task_status IN ('pending', 'running')
         ORDER BY created_at ASC LIMIT 100
        """
    )

    states = await db.fetch_all(
        """
        SELECT task_status::text AS status, artifact_status::text AS artifact,
               COUNT(*) AS n
          FROM tasks GROUP BY task_status, artifact_status ORDER BY n DESC
        """
    )

    return {
        "rabbitmq": broker,
        "outbox": {"pending": pending_outbox, "recent": recent_outbox},
        "slurm": {
            "jobs": slurm,
            # sinfo/sacct run inside the poller's container, which holds the
            # only SSH credentials. Surfacing them here means agreeing an
            # interface with that module rather than opening a second connection
            # to the cluster from the web layer.
            "cluster_probe": {
                "available": False,
                "reason": "sinfo/sacct są dostępne wyłącznie z kontenera pollera "
                          "(tam są poświadczenia SSH). Do wystawienia przez "
                          "uzgodniony endpoint pollera, nie przez drugie "
                          "połączenie z klastrem.",
            },
        },
        "orphans": orphans,
        "states": states,
    }


@router.get("/submissions")
async def all_submissions(_: CurrentUser = Depends(require_admin)) -> dict:
    rows = await db.fetch_all(
        """
        SELECT s.submission_id, s.display_name, s.kind, s.builtin_name,
               s.status::text AS status, s.family::text AS family,
               s.source_sha256, s.created_at, u.email AS submitter
          FROM submissions s JOIN users u ON u.id = s.submitted_by
         ORDER BY s.created_at DESC LIMIT 200
        """
    )
    return {"submissions": rows}


@router.post("/submissions/{submission_id}/revoke")
async def revoke_submission(
    submission_id: str, _: CurrentUser = Depends(require_admin)
) -> dict:
    await db.execute(
        "UPDATE submissions SET status = 'rejected' WHERE submission_id = %s",
        (submission_id,),
    )
    return {"ok": True}
