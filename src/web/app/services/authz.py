"""Read and write authorization for runs, decided in one place.

PUBLIC_RESULTS switches the whole deployment between public and owner-only
results without touching any endpoint.
"""

from typing import Optional
from uuid import UUID

from app.security import CurrentUser
from app.settings import settings


def can_read_run(user: Optional[CurrentUser], submitted_by: UUID) -> bool:
    if user is not None:
        if user.is_admin:
            return True
        if user.id == submitted_by:
            return True
    return settings.public_results


def can_write_run(user: Optional[CurrentUser], submitted_by: UUID) -> bool:
    if user is None:
        return False
    return user.is_admin or user.id == submitted_by
