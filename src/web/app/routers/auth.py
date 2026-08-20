"""Sessions, registration and API tokens.

Every user mutation goes through the frontend's existing repository module
rather than a second implementation of the same rules -- see legacy_auth for why
that is deliberate.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, EmailStr, Field

from app import db, legacy_auth
from app.security import (
    CurrentUser,
    clear_session,
    generate_api_token,
    issue_session,
    optional_user,
    require_user,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class Registration(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: Optional[str] = None
    associated_organisation: Optional[str] = None
    associated_org_email: Optional[str] = None
    join_reason: Optional[str] = None


class TokenRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


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
async def me(user: Optional[CurrentUser] = Depends(optional_user)) -> dict:
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


@router.get("/tokens")
async def list_tokens(user: CurrentUser = Depends(require_user)) -> dict:
    rows = await db.fetch_all(
        """
        SELECT token_id, name, prefix, created_at, last_used_at, revoked_at
          FROM api_tokens WHERE user_id = %s ORDER BY created_at DESC
        """,
        (user.id,),
    )
    return {"tokens": rows}


@router.post("/tokens", status_code=status.HTTP_201_CREATED)
async def create_token(
    payload: TokenRequest, user: CurrentUser = Depends(require_user)
) -> dict:
    raw, digest, prefix = generate_api_token()
    row = await db.fetch_one(
        """
        INSERT INTO api_tokens (user_id, name, token_sha256, prefix)
        VALUES (%s, %s, %s, %s)
        RETURNING token_id, name, prefix, created_at
        """,
        (user.id, payload.name, digest, prefix),
    )
    # Returned once and never stored in plaintext.
    return {**dict(row or {}), "token": raw}


@router.delete("/tokens/{token_id}")
async def revoke_token(token_id: str, user: CurrentUser = Depends(require_user)) -> dict:
    await db.execute(
        """
        UPDATE api_tokens SET revoked_at = NOW()
         WHERE token_id = %s AND user_id = %s AND revoked_at IS NULL
        """,
        (token_id, user.id),
    )
    return {"ok": True}
