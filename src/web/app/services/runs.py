"""Reads over tasks joined with their results."""

from typing import Any, Optional
from uuid import UUID

from app import db
from app.services import naming

RUN_COLUMNS = """
    t.task_id, t.run_name, t.dataset, t.model_name, t.optimizer_name,
    t.family::text        AS family,
    t.suite::text         AS suite,
    t.task_status::text   AS task_status,
    t.artifact_status::text AS artifact_status,
    t.executor_task_id, t.executor_name, t.queue_name, t.seed, t.stop_condition,
    t.created_at, t.updated_at, t.queued_at, t.started_at, t.completed_at,
    t.error_message, t.submitted_by, t.artifact_bytes, t.artifact_files,
    t.runner_version, t.gpu_model, t.submission_id,
    u.display_name AS submitter_name,
    u.email        AS submitter_email,
    r.final_loss, r.final_accuracy, r.gradient_count, r.database_reaches,
    r.total_steps, r.total_epochs, r.wall_time_seconds,
    r.stop_reason::text AS stop_reason
"""

_FROM = """
  FROM tasks t
  JOIN users u ON u.id = t.submitted_by
  LEFT JOIN results r ON r.task_id = t.task_id
"""


def serialise(row: dict) -> dict:
    """Shape one run for the API, including the derived state label.

    The state and its wording are computed here rather than in the browser so
    that the API, the CSV export and the interface cannot describe the same run
    differently.
    """
    state = naming.derive_state(row)
    stop_reason = row.get("stop_reason")
    return {
        "task_id": str(row["task_id"]),
        "run_name": row.get("run_name"),
        "dataset": row.get("dataset"),
        "model": row.get("model_name"),
        "optimizer": row.get("optimizer_name"),
        "family": row.get("family"),
        "suite": row.get("suite"),
        "state": state,
        "state_label": naming.RUN_STATES[state]["label"],
        "state_detail": naming.RUN_STATES[state]["detail"],
        "state_tone": naming.RUN_STATES[state]["tone"],
        "task_status": row.get("task_status"),
        "artifact_status": row.get("artifact_status"),
        "artifact_bytes": row.get("artifact_bytes"),
        "artifact_files": row.get("artifact_files"),
        "slurm_job_id": row.get("executor_task_id"),
        "executor": row.get("executor_name"),
        "queue_name": row.get("queue_name"),
        "seed": row.get("seed"),
        "stop_condition": row.get("stop_condition"),
        "submitted_by": str(row["submitted_by"]),
        "submitter_name": row.get("submitter_name") or row.get("submitter_email"),
        "submission_id": str(row["submission_id"]) if row.get("submission_id") else None,
        "created_at": row.get("created_at"),
        "queued_at": row.get("queued_at"),
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "updated_at": row.get("updated_at"),
        "error_message": row.get("error_message"),
        "runner_version": row.get("runner_version"),
        "gpu_model": row.get("gpu_model"),
        "metrics": {
            "final_loss": row.get("final_loss"),
            "final_accuracy": row.get("final_accuracy"),
            "gradient_count": row.get("gradient_count"),
            "database_reaches": row.get("database_reaches"),
            "total_steps": row.get("total_steps"),
            "total_epochs": row.get("total_epochs"),
            "wall_time_seconds": row.get("wall_time_seconds"),
        } if row.get("gradient_count") is not None else None,
        "stop_reason": stop_reason,
        "stop_reason_label": (
            naming.STOP_REASONS.get(stop_reason, {}).get("label") if stop_reason else None
        ),
        "converged": (
            naming.STOP_REASONS.get(stop_reason, {}).get("converged") if stop_reason else None
        ),
    }


async def get(task_id: UUID) -> Optional[dict]:
    return await db.fetch_one(
        f"SELECT {RUN_COLUMNS} {_FROM} WHERE t.task_id = %s", (task_id,)
    )


async def listing(
    mine_for: Optional[UUID] = None,
    status: Optional[str] = None,
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    family: Optional[str] = None,
    suite: Optional[str] = None,
    optimizer: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    where: list[str] = []
    params: list[Any] = []

    if mine_for is not None:
        where.append("t.submitted_by = %s")
        params.append(mine_for)
    if status:
        where.append("t.task_status::text = %s")
        params.append(status)
    if dataset:
        where.append("t.dataset = %s")
        params.append(dataset)
    if model:
        where.append("t.model_name = %s")
        params.append(model)
    if family:
        where.append("t.family::text = %s")
        params.append(family)
    if suite:
        where.append("t.suite::text = %s")
        params.append(suite)
    if optimizer:
        where.append("t.optimizer_name = %s")
        params.append(optimizer)
    if search:
        where.append("(t.run_name ILIKE %s OR t.task_id::text ILIKE %s)")
        params.extend([f"%{search}%", f"%{search}%"])

    clause = (" WHERE " + " AND ".join(where)) if where else ""

    total_row = await db.fetch_one(
        f"SELECT COUNT(*) AS n {_FROM}{clause}", params
    )
    total = int(total_row["n"]) if total_row else 0

    rows = await db.fetch_all(
        f"""SELECT {RUN_COLUMNS} {_FROM}{clause}
            ORDER BY t.created_at DESC
            LIMIT %s OFFSET %s""",
        [*params, limit, offset],
    )
    return rows, total


async def transitions(task_id: UUID) -> list[dict]:
    return await db.fetch_all(
        """
        SELECT from_status, to_status, actor, detail, occurred_at
          FROM task_state_transitions
         WHERE task_id = %s
         ORDER BY occurred_at ASC, id ASC
        """,
        (task_id,),
    )


async def series_row(task_id: UUID) -> Optional[dict]:
    return await db.fetch_one(
        """
        SELECT epochs, loss, accuracy, gradient_count, database_reaches,
               wall_time_seconds
          FROM result_series WHERE task_id = %s
        """,
        (task_id,),
    )


async def filter_options() -> dict:
    rows = await db.fetch_all(
        """
        SELECT DISTINCT dataset, model_name, optimizer_name,
               family::text AS family, suite::text AS suite,
               task_status::text AS task_status
          FROM tasks
        """
    )
    return {
        "datasets": sorted({r["dataset"] for r in rows if r["dataset"]}),
        "models": sorted({r["model_name"] for r in rows if r["model_name"]}),
        "optimizers": sorted({r["optimizer_name"] for r in rows if r["optimizer_name"]}),
        "families": sorted({r["family"] for r in rows if r["family"]}),
        "suites": sorted({r["suite"] for r in rows if r["suite"]}),
        "statuses": sorted({r["task_status"] for r in rows if r["task_status"]}),
    }
