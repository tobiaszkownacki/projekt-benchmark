"""Submitting an optimizer: validate, record, queue.

The whole point of validating before queueing is that a broken submission which
reaches the cluster spends grant hours, and grant hours are the scarce resource.
Thirty seconds of local CPU is the cheaper place to find out.

Nothing here publishes to the broker. The task row and the queue message are
written in one transaction and drained by a separate process, so a broker outage
cannot lose a submission and the API needs no broker credentials.
"""

from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from app import db
from app.security import CurrentUser, optional_user, require_verified
from app.services import outbox, validator
from app.services.authz import can_read_run
from app.settings import settings

router = APIRouter(prefix="/api/submissions", tags=["submissions"])

BUILTIN_FAMILIES = {
    "adam": "gradient", "adamw": "gradient", "lion": "gradient",
    "rmsprop": "gradient", "sgd": "gradient", "sgd_momentum": "gradient",
    "cma-es": "gradient_free", "de": "gradient_free", "des": "gradient_free",
}


class SubmissionRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    kind: str = Field(pattern="^(builtin|uploaded)$")
    builtin_name: Optional[str] = None
    source_code: Optional[str] = None
    dataset: str = Field(min_length=1)
    model: str = Field(min_length=1)
    suite: str = Field(default="test", pattern="^(test|final)$")
    seeds: list[int] = Field(default_factory=lambda: [2137])
    max_gradient_count: Optional[int] = Field(default=None, ge=1)
    max_database_reaches: Optional[int] = Field(default=None, ge=1)
    max_epochs: Optional[int] = Field(default=None, ge=1)


async def _remaining_quota(user_id) -> int:
    row = await db.fetch_one(
        """
        SELECT COUNT(*) AS n FROM tasks
         WHERE submitted_by = %s AND created_at::date = %s
        """,
        (user_id, date.today()),
    )
    used = int(row["n"]) if row else 0
    return max(settings.daily_submission_limit - used, 0)


@router.get("/quota")
async def quota(user: CurrentUser = Depends(require_verified)) -> dict:
    remaining = await _remaining_quota(user.id)
    return {
        "limit": settings.daily_submission_limit,
        "remaining": remaining,
        "used": settings.daily_submission_limit - remaining,
        # D7 is open. The API reports a remainder in a shape that survives
        # whichever model the team picks; only the number behind it changes.
        "model": "daily",
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def submit(
    payload: SubmissionRequest, user: CurrentUser = Depends(require_verified)
) -> dict:
    if payload.kind == "builtin" and not payload.builtin_name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Brak nazwy optymalizatora")
    if payload.kind == "uploaded" and not payload.source_code:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Brak kodu optymalizatora")

    stop_condition: dict[str, Any] = {
        k: v for k, v in {
            "max_gradient_count": payload.max_gradient_count,
            "max_database_reaches": payload.max_database_reaches,
            "max_epochs": payload.max_epochs,
        }.items() if v is not None
    }
    if not stop_condition:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Podaj co najmniej jeden warunek stopu",
        )

    seeds = payload.seeds[:16] or [2137]
    remaining = await _remaining_quota(user.id)
    if len(seeds) > remaining:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Dzienny limit zgłoszeń wyczerpany. Pozostało {remaining} "
            f"z {settings.daily_submission_limit}.",
            headers={"Retry-After": "3600"},
        )

    if payload.kind == "uploaded":
        outcome = await validator.validate_source(payload.source_code or "")
        family = outcome.family
        digest = validator.sha256(payload.source_code or "")
    else:
        outcome = validator.ValidationResult(
            ok=True,
            log=f"Optymalizator wbudowany '{payload.builtin_name}' — "
                f"walidacja protokołu pominięta, kod pochodzi z repozytorium.",
            family=BUILTIN_FAMILIES.get(payload.builtin_name or "", "gradient"),
            version="builtin",
        )
        family = outcome.family
        digest = None

    async with db.connection() as conn:
        row = await (await conn.execute(
            """
            INSERT INTO submissions (
                submitted_by, display_name, kind, builtin_name, source_code,
                source_sha256, family, status, validator_log, validator_version,
                validated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            RETURNING submission_id, status::text AS status
            """,
            (
                user.id, payload.display_name, payload.kind, payload.builtin_name,
                payload.source_code, digest, family,
                "accepted" if outcome.ok else "rejected",
                outcome.log, outcome.version,
            ),
        )).fetchone()

        submission_id = row["submission_id"]

        if not outcome.ok:
            # Rejected submissions never reach the queue, but they are kept:
            # the participant needs the log, and §11.3 names "rejected at
            # validation" as a state the interface must be able to show.
            await conn.commit()
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "submission_id": str(submission_id),
                    "status": "rejected",
                    "validator_log": outcome.log,
                },
            )

        optimizer_name = payload.builtin_name or payload.display_name
        created: list[str] = []
        for seed in seeds:
            task = await (await conn.execute(
                """
                INSERT INTO tasks (
                    queue_name, executor_name, submitted_by, dataset, run_name,
                    optimizer_params, submission_id, seed, suite, model_name,
                    optimizer_name, family, stop_condition, queued_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING task_id
                """,
                (
                    settings.worker_queue, "athena", user.id, payload.dataset,
                    f"{optimizer_name}-{payload.dataset}-s{seed}",
                    Jsonb({"optimizer": optimizer_name, "seed": seed}),
                    submission_id, seed, payload.suite, payload.model,
                    optimizer_name, family, Jsonb(stop_condition),
                ),
            )).fetchone()

            task_id = task["task_id"]
            await outbox.enqueue(conn, outbox.task_message(task_id, settings.worker_queue))
            created.append(str(task_id))

        await conn.commit()

    return {
        "submission_id": str(submission_id),
        "status": "accepted",
        "validator_log": outcome.log,
        "validator_version": outcome.version,
        "family": family,
        "task_ids": created,
        "remaining_today": remaining - len(seeds),
    }


@router.get("/{submission_id}")
async def get_submission(
    submission_id: str, user: Optional[CurrentUser] = Depends(optional_user)
) -> dict:
    row = await db.fetch_one(
        """
        SELECT s.*, s.status::text AS status_text, s.family::text AS family_text
          FROM submissions s WHERE s.submission_id = %s
        """,
        (submission_id,),
    )
    if row is None or not can_read_run(user, row["submitted_by"]):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")

    tasks = await db.fetch_all(
        "SELECT task_id FROM tasks WHERE submission_id = %s ORDER BY seed",
        (submission_id,),
    )
    return {
        "submission_id": str(row["submission_id"]),
        "display_name": row["display_name"],
        "kind": row["kind"],
        "builtin_name": row["builtin_name"],
        "source_sha256": row["source_sha256"],
        "family": row["family_text"],
        "status": row["status_text"],
        "validator_log": row["validator_log"],
        "validator_version": row["validator_version"],
        "created_at": row["created_at"],
        "task_ids": [str(t["task_id"]) for t in tasks],
        # Source is returned only to people who may already read the run, and it
        # is text for display. It is never imported and never executed.
        "source_code": row["source_code"],
    }


@router.get("")
async def list_submissions(user: CurrentUser = Depends(require_verified)) -> dict:
    rows = await db.fetch_all(
        """
        SELECT submission_id, display_name, kind, builtin_name,
               status::text AS status, family::text AS family, created_at
          FROM submissions
         WHERE submitted_by = %s OR %s
         ORDER BY created_at DESC LIMIT 200
        """,
        (user.id, user.is_admin),
    )
    return {"submissions": rows}
