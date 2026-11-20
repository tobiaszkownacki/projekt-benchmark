"""Read and write authorization for runs, decided in one place.

A run is private to the account that submitted it; administrators see every
run. There is no reference or baseline category in the schema yet, so no run
is visible to everyone -- when one is added (PROD-003), it belongs here.
"""

from uuid import UUID

from app.security import CurrentUser


def can_read_run(user: CurrentUser | None, submitted_by: UUID) -> bool:
    if user is None:
        return False
    return user.is_admin or user.id == submitted_by


def can_write_run(user: CurrentUser | None, submitted_by: UUID) -> bool:
    if user is None:
        return False
    return user.is_admin or user.id == submitted_by


def visible_owner(user: CurrentUser) -> UUID | None:
    """The submitter a listing must be restricted to, or None for everything."""
    return None if user.is_admin else user.id
