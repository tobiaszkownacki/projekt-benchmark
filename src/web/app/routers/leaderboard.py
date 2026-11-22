"""Leaderboard and overview counters."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app import db
from app.security import CurrentUser, optional_user
from app.services import leaderboard as leaderboard_service
from app.services import series as series_service

router = APIRouter(prefix="/api", tags=["leaderboard"])


@router.get("/leaderboard")
async def leaderboard(
    dataset: str | None = None,
    model: str | None = None,
    family: str | None = None,
    suite: str | None = None,
    score: str = Query(leaderboard_service.DEFAULT_FORMULA),
    limit: int = Query(200, ge=1, le=500),
    user: CurrentUser | None = Depends(optional_user),
) -> dict:
    return await leaderboard_service.query(
        dataset=dataset,
        model=model,
        family=family,
        suite=suite,
        score=score,
        limit=limit,
        user=user,
    )


@router.get("/leaderboard/curves")
async def leaderboard_curves(
    dataset: str | None = None,
    model: str | None = None,
    suite: str | None = None,
    x: str = Query("database_reaches"),
    metric: str = Query("loss"),
    points: int = Query(160, ge=20, le=400),
) -> dict:
    """Public, aggregate-only convergence curves; see the service docstring."""
    if x not in series_service.X_AXES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown x axis")
    if metric not in series_service.METRICS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown metric")
    return await leaderboard_service.public_curves(
        dataset=dataset,
        model=model,
        suite=suite,
        x=x,
        metric=metric,
        points=points,
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
            (SELECT COUNT(*) FROM tasks WHERE task_status = 'COMPLETED')        AS completed,
            (SELECT COUNT(*) FROM tasks WHERE task_status = 'FAILED')           AS failed,
            (SELECT COUNT(*) FROM tasks
              WHERE task_status IN ('PENDING', 'SUBMITTED', 'RUNNING'))                      AS active,
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
