"""Run listing, detail, convergence series and state history."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.security import CurrentUser, optional_user
from app.services import runs as runs_service
from app.services import series as series_service
from app.services.authz import can_read_run

router = APIRouter(prefix="/api/runs", tags=["runs"])


async def _load_visible(task_id: UUID, user: Optional[CurrentUser]) -> dict:
    row = await runs_service.get(task_id)
    # 404 rather than 403 in both branches: a 403 confirms that a run with this
    # identifier exists, which is information the requester has not earned.
    if row is None or not can_read_run(user, row["submitted_by"]):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return row


@router.get("")
async def list_runs(
    mine: bool = False,
    status_filter: Optional[str] = Query(None, alias="status"),
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    family: Optional[str] = None,
    suite: Optional[str] = None,
    optimizer: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: Optional[CurrentUser] = Depends(optional_user),
) -> dict:
    if mine and user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")

    mine_for = user.id if (mine and user) else None
    rows, total = await runs_service.listing(
        mine_for=mine_for,
        status=status_filter,
        dataset=dataset,
        model=model,
        family=family,
        suite=suite,
        optimizer=optimizer,
        search=search,
        limit=limit,
        offset=offset,
    )
    visible = [r for r in rows if can_read_run(user, r["submitted_by"])]
    return {
        "runs": [runs_service.serialise(r) for r in visible],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/filters")
async def run_filters() -> dict:
    return await runs_service.filter_options()


@router.get("/{task_id}")
async def get_run(
    task_id: UUID, user: Optional[CurrentUser] = Depends(optional_user)
) -> dict:
    row = await _load_visible(task_id, user)
    payload = runs_service.serialise(row)
    payload["can_manage"] = bool(
        user and (user.is_admin or user.id == row["submitted_by"])
    )
    return payload


@router.get("/{task_id}/transitions")
async def get_transitions(
    task_id: UUID, user: Optional[CurrentUser] = Depends(optional_user)
) -> dict:
    await _load_visible(task_id, user)
    return {"transitions": await runs_service.transitions(task_id)}


@router.get("/{task_id}/series")
async def get_series(
    task_id: UUID,
    x: str = Query("gradient_count"),
    metric: str = Query("loss"),
    points: int = Query(series_service.DEFAULT_POINTS, ge=10, le=series_service.MAX_POINTS),
    user: Optional[CurrentUser] = Depends(optional_user),
) -> dict:
    await _load_visible(task_id, user)

    if x not in series_service.X_AXES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown x axis")
    if metric not in series_service.METRICS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown metric")

    row = await runs_service.series_row(task_id)
    if row is None:
        return {
            "task_id": str(task_id),
            "x": x,
            "metric": metric,
            "points": [],
            "truncated": False,
            "original_points": 0,
        }

    pairs = series_service.series_points(row, x, metric)
    sampled, truncated = series_service.downsample_pairs(pairs, points)
    return {
        "task_id": str(task_id),
        "x": x,
        "metric": metric,
        "points": [[p[0], p[1]] for p in sampled],
        # Reported so the reader can tell they are looking at an approximation.
        "truncated": truncated,
        "downsample": "lttb" if truncated else "none",
        "original_points": len(pairs),
    }
