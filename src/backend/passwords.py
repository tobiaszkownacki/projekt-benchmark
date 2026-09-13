"""Password hashing, with no database dependency.

Separate from ``legacy_auth`` on purpose: the seeder needs ``hash_password`` and
nothing else, and importing it from a module that reaches for the async
connection pool would pull psycopg_pool into a script that opens its own plain
``psycopg.connect``. One definition, so a hash written by the seeder is one the
login path verifies.
"""

import bcrypt


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def validate_password_strength(password: str) -> str | None:
    if len(password) < 8:
        return "Hasło musi mieć co najmniej 8 znaków."
    return None
