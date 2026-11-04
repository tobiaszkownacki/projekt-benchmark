"""Test environment for the pipeline side.

The pipeline is imported by its top-level package names (pipeline, shared) both
in the containers and here, so src/ has to be on the path; it holds no
installable distribution of its own.
"""

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MIGRATIONS = Path(__file__).resolve().parents[2] / "db/migrations"


@pytest.fixture(scope="session")
def database_url() -> str:
    """A database at the current schema, or a skip without TEST_DATABASE_URL.

    Migrated into the same schema_migrations table the web runner keeps, so the
    two suites can share one database and run in either order.
    """
    url = os.environ.get("TEST_DATABASE_URL", "")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    import psycopg

    with psycopg.connect(url) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
        )
        known = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        for path in sorted(MIGRATIONS.glob("*.sql")):
            if path.stem not in known:
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
    return url


@pytest.fixture
def sql_repository(database_url):
    import psycopg

    from pipeline.sql_task_repository import SqlTaskRepository
    from shared.connectors.postgres import PostGresConnector

    class TestConnector(PostGresConnector):
        def __enter__(self) -> "TestConnector":
            self.conn = psycopg.connect(database_url)
            return self

    return SqlTaskRepository(TestConnector)


@pytest.fixture
def new_task(database_url):
    """Inserts a task in the given state and returns its id."""
    import psycopg

    def insert(task_status: str = "PENDING", executor_task_id: str | None = None) -> str:
        with psycopg.connect(database_url, autocommit=True) as conn:
            user_id = conn.execute(
                "INSERT INTO users (email, auth_provider, role, password_hash) "
                "VALUES (%s, 'email', 'verified', 'not-a-usable-hash') RETURNING id",
                (f"pipeline-{uuid.uuid4()}@example.test",),
            ).fetchone()[0]
            task_id = conn.execute(
                "INSERT INTO tasks (queue_name, executor_name, submitted_by, dataset, run_name, task_status, "
                "executor_task_id) VALUES ('q', 'test', %s, 'wine_quality', 'run', %s, %s) RETURNING task_id",
                (user_id, task_status, executor_task_id),
            ).fetchone()[0]
        return str(task_id)

    return insert
