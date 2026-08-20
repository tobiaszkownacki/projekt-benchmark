"""Leaderboard and overview counters."""

from typing import Optional

from fastapi import APIRouter, Query

from app import db
from app.services import leaderboard as leaderboard_service

router = APIRouter(prefix="/api", tags=["leaderboard"])


@router.get("/leaderboard")
async def leaderboard(
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    family: Optional[str] = None,
    suite: Optional[str] = None,
    score: str = Query(leaderboard_service.DEFAULT_FORMULA),
    limit: int = Query(200, ge=1, le=500),
) -> dict:
    return await leaderboard_service.query(
        dataset=dataset, model=model, family=family, suite=suite,
        score=score, limit=limit,
    )


@router.get("/leaderboard/filters")
async def leaderboard_filters() -> dict:
    return await leaderboard_service.filter_options()


@router.get("/overview")
async def overview() -> dict:
    """Counters for the landing page.

    GPU hours are reported as accumulated wall time and labelled as such. It is
    the honest number available: the local backend runs on CPU, and presenting
    its seconds as cluster GPU hours would be a small lie in the one place a
    visitor forms their first impression.
    """
    row = await db.fetch_one(
        """
        SELECT
            (SELECT COUNT(*) FROM submissions)                                  AS submissions,
            (SELECT COUNT(*) FROM tasks)                                        AS runs,
            (SELECT COUNT(*) FROM tasks WHERE task_status = 'completed')        AS completed,
            (SELECT COUNT(*) FROM tasks WHERE task_status = 'failed')           AS failed,
            (SELECT COUNT(*) FROM tasks
              WHERE task_status IN ('pending', 'running'))                      AS active,
            (SELECT COUNT(DISTINCT submitted_by) FROM tasks)                    AS participants,
            (SELECT COUNT(DISTINCT dataset) FROM tasks WHERE dataset IS NOT NULL) AS datasets,
            (SELECT COUNT(DISTINCT optimizer_name) FROM tasks
              WHERE optimizer_name IS NOT NULL)                                 AS optimizers,
            (SELECT COALESCE(SUM(wall_time_seconds), 0) FROM results)           AS compute_seconds,
            (SELECT COALESCE(SUM(gradient_count), 0) FROM results)              AS gradients,
            (SELECT COALESCE(SUM(database_reaches), 0) FROM results)            AS samples
        """
    )
    return dict(row or {})
