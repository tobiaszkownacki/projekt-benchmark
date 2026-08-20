"""Session cookies for the browser, bearer tokens for CLI and CI.

Both paths resolve to the same user object and the same authorisation checks --
§15 requires programmatic submission, and a second, parallel permission model
would be the obvious place for the two to drift apart.
"""

import hashlib
import secrets
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Request, Response, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app import db
from app.settings import settings

_TOKEN_PREFIX = "bmk_"


@dataclass(frozen=True)
class CurrentUser:
    id: UUID
    email: str
    role: str
    display_name: Optional[str]
    is_active: bool
    has_join_info: bool

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_verified(self) -> bool:
        return self.role in ("verified", "admin")


def _serializer() -> URLSafeTimedSerializer:
    if not settings.session_secret:
        raise RuntimeError("SESSION_SECRET is not configured")
    return URLSafeTimedSerializer(settings.session_secret, salt="benchmark-session")


def issue_session(response: Response, user_id: UUID) -> None:
    token = _serializer().dumps(str(user_id))
    response.set_cookie(
        settings.session_cookie,
        token,
        max_age=settings.session_max_age,
        httponly=True,
        samesite="strict",
        secure=settings.secure_cookies,
        path="/",
    )


def clear_session(response: Response) -> None:
    response.delete_cookie(settings.session_cookie, path="/")


def _read_session(request: Request) -> Optional[UUID]:
    raw = request.cookies.get(settings.session_cookie)
    if not raw:
        return None
    try:
        value = _serializer().loads(raw, max_age=settings.session_max_age)
        return UUID(value)
    except (BadSignature, SignatureExpired, ValueError):
        return None


def generate_api_token() -> tuple[str, str, str]:
    """Return (plaintext, sha256, prefix). Only the digest is ever stored."""
    raw = _TOKEN_PREFIX + secrets.token_urlsafe(32)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return raw, digest, raw[: len(_TOKEN_PREFIX) + 6]


async def _user_from_bearer(request: Request) -> Optional[dict]:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    presented = header.split(" ", 1)[1].strip()
    if not presented:
        return None
    digest = hashlib.sha256(presented.encode()).hexdigest()
    row = await db.fetch_one(
        """
        SELECT u.id, u.email, u.role, u.display_name, u.is_active,
               u.join_reason, u.associated_organisation, t.token_id
          FROM api_tokens t
          JOIN users u ON u.id = t.user_id
         WHERE t.token_sha256 = %s AND t.revoked_at IS NULL
        """,
        (digest,),
    )
    if row is None:
        return None
    await db.execute(
        "UPDATE api_tokens SET last_used_at = NOW() WHERE token_id = %s",
        (row["token_id"],),
    )
    return row


async def _user_from_cookie(request: Request) -> Optional[dict]:
    user_id = _read_session(request)
    if user_id is None:
        return None
    return await db.fetch_one(
        """
        SELECT id, email, role, display_name, is_active,
               join_reason, associated_organisation
          FROM users WHERE id = %s
        """,
        (user_id,),
    )


async def optional_user(request: Request) -> Optional[CurrentUser]:
    row = await _user_from_bearer(request) or await _user_from_cookie(request)
    if row is None or not row["is_active"]:
        return None
    return CurrentUser(
        id=row["id"],
        email=row["email"],
        role=row["role"],
        display_name=row["display_name"],
        is_active=row["is_active"],
        has_join_info=bool(row.get("join_reason") or row.get("associated_organisation")),
    )


async def require_user(
    user: Optional[CurrentUser] = Depends(optional_user),
) -> CurrentUser:
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
    return user


async def require_verified(
    user: CurrentUser = Depends(require_user),
) -> CurrentUser:
    if not user.is_verified:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Your account is waiting for approval by an administrator",
        )
    return user


async def require_admin(user: CurrentUser = Depends(require_user)) -> CurrentUser:
    if not user.is_admin:
        # Deliberately 404: a 403 on an admin path confirms the path exists.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    return user
