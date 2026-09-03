"""Bridge to the authentication code in ``src/frontend``.

Reuses ``auth/repository.py`` rather than reimplementing password hashing, the
OAuth upsert rules and the approval flow.

Those modules are synchronous. FastAPI runs plain ``def`` dependencies in a
worker thread anyway, so the calls here go through ``asyncio.to_thread`` and the
blocking pool never touches the event loop. User mutations are rare; the hot
read paths use the async pool directly.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if _FRONTEND_DIR.is_dir() and str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))

from auth import passwords as _passwords          # noqa: E402
from auth import repository as _repository        # noqa: E402

User = _repository.User

hash_password = _passwords.hash_password
verify_password = _passwords.verify_password
validate_password_strength = _passwords.validate_password_strength


async def get_by_email(email: str) -> Optional[User]:
    return await asyncio.to_thread(_repository.get_by_email, email)


async def get_by_id(user_id: UUID) -> Optional[User]:
    return await asyncio.to_thread(_repository.get_by_id, user_id)


async def create_email_user(**kwargs: Any) -> User:
    return await asyncio.to_thread(lambda: _repository.create_email_user(**kwargs))


async def upsert_oauth_user(**kwargs: Any) -> User:
    return await asyncio.to_thread(lambda: _repository.upsert_oauth_user(**kwargs))


async def set_join_info(user_id: UUID, **kwargs: Any) -> Optional[User]:
    return await asyncio.to_thread(lambda: _repository.set_join_info(user_id, **kwargs))


async def update_last_login(user_id: UUID) -> None:
    await asyncio.to_thread(_repository.update_last_login, user_id)


async def list_unverified() -> list[User]:
    return await asyncio.to_thread(_repository.list_unverified)


async def approve_user(user_id: UUID) -> User:
    return await asyncio.to_thread(_repository.approve_user, user_id)
