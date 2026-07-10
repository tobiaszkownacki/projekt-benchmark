from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from psycopg.rows import dict_row

from auth.database import get_connection
from auth.passwords import hash_password


@dataclass
class User:
    id: UUID
    email: str
    role: str
    auth_provider: str
    display_name: Optional[str]
    is_active: bool
    created_at: datetime
    last_login_at: Optional[datetime]
    associated_organisation: Optional[str] = None
    associated_org_email: Optional[str] = None
    join_reason: Optional[str] = None
    password_hash: Optional[str] = None


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


def has_join_info(user: User) -> bool:
    return bool(user.join_reason or user.associated_organisation)


def get_by_id(user_id: UUID) -> Optional[User]:
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
            return _row_to_user(row) if row else None


def get_by_email(email: str) -> Optional[User]:
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email.lower(),))
            row = cur.fetchone()
            return _row_to_user(row) if row else None


def create_email_user(
    email: str,
    password: str,
    display_name: Optional[str] = None,
    associated_organisation: Optional[str] = None,
    associated_org_email: Optional[str] = None,
    join_reason: Optional[str] = None,
) -> User:
    email_lower = email.lower()
    password_hash = hash_password(password)
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO users (
                    email, password_hash, role, auth_provider, display_name,
                    associated_organisation, associated_org_email, join_reason
                )
                VALUES (%s, %s, 'unverified', 'email', %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    email_lower,
                    password_hash,
                    display_name,
                    associated_organisation,
                    associated_org_email,
                    join_reason,
                ),
            )
            row = cur.fetchone()
            assert row is not None
            return _row_to_user(row)


def set_join_info(
    user_id: UUID,
    associated_organisation: Optional[str] = None,
    associated_org_email: Optional[str] = None,
    join_reason: Optional[str] = None,
) -> Optional[User]:
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE users
                SET associated_organisation = %s,
                    associated_org_email = %s,
                    join_reason = %s
                WHERE id = %s
                RETURNING *
                """,
                (associated_organisation, associated_org_email, join_reason, user_id),
            )
            row = cur.fetchone()
            return _row_to_user(row) if row else None


def upsert_oauth_user(
    email: str,
    oauth_sub: str,
    auth_provider: str,
    display_name: Optional[str] = None,
) -> User:
    email_lower = email.lower()
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email_lower,))
            existing = cur.fetchone()

            if existing:
                if not existing["is_active"]:
                    return _row_to_user(existing)
                cur.execute(
                    """
                    UPDATE users
                    SET last_login_at = NOW(),
                        display_name = COALESCE(%s, display_name),
                        oauth_sub = COALESCE(%s, oauth_sub)
                    WHERE id = %s
                    RETURNING *
                    """,
                    (display_name, oauth_sub, existing["id"]),
                )
                row = cur.fetchone()
                assert row is not None
                return _row_to_user(row)

            cur.execute(
                """
                INSERT INTO users (email, role, auth_provider, oauth_sub, display_name, last_login_at)
                VALUES (%s, 'unverified', %s, %s, %s, NOW())
                RETURNING *
                """,
                (email_lower, auth_provider, oauth_sub, display_name),
            )
            row = cur.fetchone()
            assert row is not None
            return _row_to_user(row)


def update_last_login(user_id: UUID) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET last_login_at = NOW() WHERE id = %s",
            (user_id,),
        )


def list_unverified() -> List[User]:
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT * FROM users
                WHERE role = 'unverified' AND is_active = TRUE
                ORDER BY created_at ASC
                """
            )
            return [_row_to_user(row) for row in cur.fetchall()]


def approve_user(user_id: UUID) -> Optional[User]:
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE users SET role = 'verified'
                WHERE id = %s AND role = 'unverified'
                RETURNING *
                """,
                (user_id,),
            )
            row = cur.fetchone()
            assert row is not None
            return _row_to_user(row) if row else None
