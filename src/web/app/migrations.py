"""Forward-only SQL migrations applied at start-up.

docker-entrypoint-initdb.d runs only against an empty volume, so a schema change
made after the database exists is silently skipped. Applying numbered files and
recording them makes the state of a live database knowable.
"""

import logging
from pathlib import Path

import psycopg

from app.settings import settings

logger = logging.getLogger(__name__)

_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def migration_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix == ".sql")


async def apply_migrations(database_url: str = "", directory: Path | None = None) -> list[str]:
    """Apply every unapplied migration in order. Returns the versions applied."""
    database_url = database_url or settings.database_url
    directory = directory or settings.migrations_dir

    files = migration_files(directory)
    if not files:
        logger.warning("No migration files found in %s", directory)
        return []

    applied: list[str] = []
    conn = await psycopg.AsyncConnection.connect(database_url, autocommit=False)
    try:
        await conn.execute(_TABLE)
        await conn.commit()

        cur = await conn.execute("SELECT version FROM schema_migrations")
        known = {row[0] for row in await cur.fetchall()}

        for path in files:
            version = path.stem
            if version in known:
                continue
            logger.info("Applying migration %s", version)
            try:
                await conn.execute(path.read_text(encoding="utf-8"))
                await conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
                )
                await conn.commit()
                applied.append(version)
            except Exception:
                await conn.rollback()
                logger.exception("Migration %s failed", version)
                raise
    finally:
        await conn.close()

    return applied
