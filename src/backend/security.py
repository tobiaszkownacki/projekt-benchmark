"""Session cookies for the browser. One user object, one set of auth checks."""

from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, HTTPException, Request, Response, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from backend import db
from backend.settings import settings


@dataclass(frozen=True)
class CurrentUser:
    id: UUID
    email: str
    role: str
    display_name: str | None
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
        path=settings.session_cookie_path,
    )


def clear_session(response: Response) -> None:
    response.delete_cookie(settings.session_cookie, path=settings.session_cookie_path)


def _read_session(request: Request) -> UUID | None:
    raw = request.cookies.get(settings.session_cookie)
    if not raw:
        return None
    try:
        value = _serializer().loads(raw, max_age=settings.session_max_age)
        return UUID(value)
    except (BadSignature, SignatureExpired, ValueError):
        return None


async def _user_from_cookie(request: Request) -> dict | None:
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


async def optional_user(request: Request) -> CurrentUser | None:
    row = await _user_from_cookie(request)
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
    user: CurrentUser | None = Depends(optional_user),
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
