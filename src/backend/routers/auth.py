"""Sessions and registration.

User mutations go through legacy_auth (password hashing, the approval flow),
not a second copy of those rules.
"""

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, EmailStr, Field

from backend import legacy_auth
from backend.security import (
    CurrentUser,
    clear_session,
    issue_session,
    optional_user,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class Registration(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: str | None = None
    associated_organisation: str | None = None
    associated_org_email: str | None = None
    join_reason: str | None = None


def _public(user: CurrentUser) -> dict:
    return {
        "id": str(user.id),
        "email": user.email,
        "role": user.role,
        "display_name": user.display_name,
        "is_admin": user.is_admin,
        "is_verified": user.is_verified,
        "has_join_info": user.has_join_info,
    }


@router.get("/me")
async def me(user: CurrentUser | None = Depends(optional_user)) -> dict:
    return {"user": _public(user) if user else None}


@router.post("/login")
async def login(payload: Credentials, response: Response) -> dict:
    account = await legacy_auth.get_by_email(payload.email)
    # One message for "no such account" and for "wrong password": distinguishing
    # them turns the login form into a way to enumerate who has registered.
    invalid = HTTPException(status.HTTP_401_UNAUTHORIZED, "Nieprawidłowy e-mail lub hasło")
    if account is None or not account.password_hash:
        raise invalid
    if not legacy_auth.verify_password(payload.password, account.password_hash):
        raise invalid
    if not account.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Konto jest nieaktywne")

    await legacy_auth.update_last_login(account.id)
    issue_session(response, account.id)
    return {
        "user": {
            "id": str(account.id),
            "email": account.email,
            "role": account.role,
            "display_name": account.display_name,
            "is_admin": account.role == "admin",
            "is_verified": account.role in ("verified", "admin"),
            "has_join_info": bool(account.join_reason or account.associated_organisation),
        }
    }


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(payload: Registration) -> dict:
    problem = legacy_auth.validate_password_strength(payload.password)
    if problem:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, problem)
    if await legacy_auth.get_by_email(payload.email):
        raise HTTPException(status.HTTP_409_CONFLICT, "Konto o tym adresie już istnieje")

    account = await legacy_auth.create_email_user(
        email=payload.email,
        password=payload.password,
        display_name=payload.display_name,
        associated_organisation=payload.associated_organisation,
        associated_org_email=payload.associated_org_email,
        join_reason=payload.join_reason,
    )
    return {
        "id": str(account.id),
        "email": account.email,
        "role": account.role,
        "note": "Konto czeka na zatwierdzenie przez administratora.",
    }


@router.post("/logout")
async def logout(response: Response) -> dict:
    clear_session(response)
    return {"ok": True}
