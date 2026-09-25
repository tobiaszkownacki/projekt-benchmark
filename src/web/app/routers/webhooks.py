"""Completion callbacks from the executor.

A job on the cluster knows it has finished long before a poller asks. The trap
around the run posts here on its way out, which turns a wait of up to one poll
interval into a wait of seconds.

The poller stays: a job killed before its script started, a node without a route
out, or a curl that never returned leave nothing to post, and sacct classifies a
failure more precisely than the shell's exit status. Both paths write only to a
task that is not finished yet, so whichever arrives second changes nothing.
"""

import logging

from fastapi import APIRouter, HTTPException, Query, Request, status
from psycopg.types.json import Jsonb

from app import db
from app.security import token_digest
from app.services import outbox
from app.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/executors", tags=["executors"])

_BEARER = {"WWW-Authenticate": "Bearer"}


def _presented_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token", headers=_BEARER)
    token = header.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token", headers=_BEARER)
    return token


@router.post("/callback")
async def callback(
    request: Request,
    job_id: str = Query(min_length=1),
    state: str = Query(pattern="^(COMPLETED|FAILED)$"),
    exit_code: int = Query(ge=0, le=255),
) -> dict:
    """Close a task the executor reports as finished.

    The token is the identity; job_id is a cross-check, because SLURM job ids
    are small sequential numbers and this endpoint is public.
    """
    digest = token_digest(_presented_token(request))

    async with db.connection() as conn:
        row = await (
            await conn.execute(
                """
                SELECT w.task_id, w.uses, t.task_status::text AS task_status, t.executor_task_id
                  FROM task_webhook_tokens w
                  JOIN tasks t USING (task_id)
                 WHERE w.token_sha256 = %s AND w.expires_at > NOW()
                """,
                (digest,),
            )
        ).fetchone()

        # 401 rather than 404 for an unknown token: 404 would confirm that a
        # token existed and merely expired.
        if row is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown or expired token", headers=_BEARER)
        if row["uses"] >= settings.webhook_token_max_uses:
            logger.warning("Callback for task_id=%s past its use ceiling", row["task_id"])
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is spent", headers=_BEARER)

        task_id = row["task_id"]
        await conn.execute(
            "UPDATE task_webhook_tokens SET uses = uses + 1, last_used_at = NOW() WHERE task_id = %s",
            (task_id,),
        )

        known_job = row["executor_task_id"]
        if known_job and known_job != job_id:
            # Something is running on the cluster that this row does not point
            # at, which is the only symptom an orphaned sbatch ever produces.
            logger.error("task_id=%s is job %s, but job %s reported for it", task_id, known_job, job_id)
            await conn.commit()
            raise HTTPException(status.HTTP_409_CONFLICT, "Job id does not match this task")

        final_status = "COMPLETED" if state == "COMPLETED" and exit_code == 0 else "FAILED"
        cursor = await conn.execute(
            """
            UPDATE tasks
               SET task_status = %s,
                   executor_task_id = COALESCE(executor_task_id, %s),
                   exit_code = %s,
                   completed_at = NOW(),
                   artifact_status = 'downloading',
                   updated_at = NOW()
             WHERE task_id = %s
               AND task_status NOT IN ('COMPLETED', 'FAILED')
            """,
            (final_status, job_id, exit_code, task_id),
        )
        applied = cursor.rowcount > 0

        if applied:
            # The trigger on tasks records the transition with actor 'system',
            # which cannot tell this path from the poller's.
            await conn.execute(
                """
                INSERT INTO task_state_transitions (task_id, to_status, actor, detail)
                VALUES (%s, %s, 'executor_callback', %s)
                """,
                (task_id, final_status, Jsonb({"job_id": job_id, "exit_code": exit_code, "state": state})),
            )
            # Artifacts are fetched for failures too: the log is the most useful
            # thing a failed run leaves behind. The poller publishes the same
            # request only for tasks it closes itself, and it closes none of
            # these.
            await outbox.enqueue(
                conn,
                {"task_id": str(task_id)},
                routing_key=settings.topology.downloader_queue,
            )

        await conn.commit()

    return {"task_id": str(task_id), "status": final_status, "applied": applied}
