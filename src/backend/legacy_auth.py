"""User accounts: password hashing and the `users` table.

Previously a bridge to ``src/frontend/auth`` (the Streamlit app). That code is
gone, so the handful of operations the auth and admin routers need are
implemented here directly against the async pool, in the same raw-SQL style as
the rest of the package.
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import bcrypt

from backend import db

# --- passwords (was src/frontend/auth/passwords.py) -------------------------


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def validate_password_strength(password: str) -> str | None:
    if len(password) < 8:
        return "Hasło musi mieć co najmniej 8 znaków."
    return None


# --- users -----------------------------------------------------------------


@dataclass
class User:
    id: UUID
    email: str
    role: str
    auth_provider: str
    display_name: str | None
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None
    associated_organisation: str | None = None
    associated_org_email: str | None = None
    join_reason: str | None = None
    password_hash: str | None = None


def _row_to_user(row: dict) -> User:
    return User(
        id=row["id"],
        email=row["email"],
        role=row["role"],
        auth_provider=row["auth_provider"],
        display_name=row["display_name"],
        is_active=row["is_active"],
        created_at=row["created_at"],
        last_login_at=row["last_login_at"],
        associated_organisation=row.get("associated_organisation"),
        associated_org_email=row.get("associated_org_email"),
        join_reason=row.get("join_reason"),
        password_hash=row.get("password_hash"),
    )


async def get_by_id(user_id: UUID) -> User | None:
    row = await db.fetch_one("SELECT * FROM users WHERE id = %s", (user_id,))
    return _row_to_user(row) if row else None


async def get_by_email(email: str) -> User | None:
    row = await db.fetch_one("SELECT * FROM users WHERE email = %s", (email.lower(),))
    return _row_to_user(row) if row else None


async def create_email_user(
    email: str,
    password: str,
    display_name: str | None = None,
    associated_organisation: str | None = None,
    associated_org_email: str | None = None,
    join_reason: str | None = None,
) -> User:
    row = await db.fetch_one(
        """
        INSERT INTO users (
            email, password_hash, role, auth_provider, display_name,
            associated_organisation, associated_org_email, join_reason
        )
        VALUES (%s, %s, 'unverified', 'email', %s, %s, %s, %s)
        RETURNING *
        """,
        (
            email.lower(),
            hash_password(password),
            display_name,
            associated_organisation,
            associated_org_email,
            join_reason,
        ),
    )
    if row is None:
        raise RuntimeError("Failed to create user: no row returned.")
    return _row_to_user(row)


async def update_last_login(user_id: UUID) -> None:
    await db.execute("UPDATE users SET last_login_at = NOW() WHERE id = %s", (user_id,))


async def list_unverified() -> list[User]:
    rows = await db.fetch_all(
        """
        SELECT * FROM users
        WHERE role = 'unverified' AND is_active = TRUE
        ORDER BY created_at ASC
        """
    )
    return [_row_to_user(row) for row in rows]


async def approve_user(user_id: UUID) -> User:
    row = await db.fetch_one(
        """
        UPDATE users SET role = 'verified'
        WHERE id = %s AND role = 'unverified'
        RETURNING *
        """,
        (user_id,),
    )
    if row is None:
        raise ValueError(f"User with id {user_id} not found or not unverified.")
    return _row_to_user(row)
