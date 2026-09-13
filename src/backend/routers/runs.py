"""Run listing, detail and state history."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.security import CurrentUser, optional_user
from backend.services import runs as runs_service
from backend.services.authz import can_read_run

router = APIRouter(prefix="/api/runs", tags=["runs"])


async def _load_visible(task_id: UUID, user: CurrentUser | None) -> dict:
    row = await runs_service.get(task_id)
    # 404 rather than 403 in both branches: a 403 confirms that a run with this
    # identifier exists, which is information the requester has not earned.
    if row is None or not can_read_run(user, row["submitted_by"]):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return row


@router.get("")
async def list_runs(
    mine: bool = False,
    status_filter: str | None = Query(None, alias="status"),
    dataset: str | None = None,
    model: str | None = None,
    family: str | None = None,
    suite: str | None = None,
    optimizer: str | None = None,
    search: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: CurrentUser | None = Depends(optional_user),
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
async def get_run(task_id: UUID, user: CurrentUser | None = Depends(optional_user)) -> dict:
    row = await _load_visible(task_id, user)
    payload = runs_service.serialise(row)
    payload["can_manage"] = bool(user and (user.is_admin or user.id == row["submitted_by"]))
    return payload


@router.get("/{task_id}/transitions")
async def get_transitions(task_id: UUID, user: CurrentUser | None = Depends(optional_user)) -> dict:
    await _load_visible(task_id, user)
    return {"transitions": await runs_service.transitions(task_id)}
