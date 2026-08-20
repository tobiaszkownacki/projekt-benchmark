"""Who may see a run.

D4 -- whether results are public or owner-only -- is not ours to close; §17 lists
it as a team decision. So the policy lives in one function behind one flag
instead of being spread across endpoints as a repeated condition. Closing D4
means changing PUBLIC_RESULTS, not auditing every route.
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
