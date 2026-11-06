"""executor_task_id is write-once in the database itself (migration 0014).

A replaced id would orphan a job still running on the cluster, so the schema
refuses it whatever code path tries.
"""

import os
import uuid

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not set")


@pytest.fixture(scope="module")
def migrated(tmp_path_factory):
    # Starting the app applies the migrations, as it does in production.
    os.environ["ARTIFACT_ROOT"] = str(tmp_path_factory.mktemp("downloads"))
    os.environ.setdefault("STATIC_ROOT", "/nonexistent")

    from app.main import app

    with TestClient(app):
        yield


@pytest.fixture
def conn(migrated):
    import psycopg

    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        yield connection


def _task(conn, executor_task_id: str | None) -> str:
    user_id = conn.execute(
        """
        INSERT INTO users (email, auth_provider, role, password_hash)
        VALUES (%s, 'email', 'verified', 'not-a-usable-hash')
        RETURNING id
        """,
        (f"write-once-{uuid.uuid4()}@example.test",),
    ).fetchone()[0]
    return conn.execute(
        """
        INSERT INTO tasks (queue_name, executor_name, submitted_by, dataset, run_name, executor_task_id)
        VALUES ('q', 'test', %s, 'wine_quality', 'run', %s)
        RETURNING task_id
        """,
        (user_id, executor_task_id),
    ).fetchone()[0]


def _job_id() -> str:
    return f"job-{uuid.uuid4().hex[:12]}"


def test_the_first_id_can_be_set(conn):
    task_id = _task(conn, None)
    job = _job_id()
    conn.execute("UPDATE tasks SET executor_task_id = %s WHERE task_id = %s", (job, task_id))
    assert conn.execute("SELECT executor_task_id FROM tasks WHERE task_id = %s", (task_id,)).fetchone()[0] == job


def test_writing_the_same_id_again_is_allowed(conn):
    job = _job_id()
    task_id = _task(conn, job)
    conn.execute("UPDATE tasks SET executor_task_id = %s WHERE task_id = %s", (job, task_id))


def test_replacing_an_id_is_refused(conn):
    import psycopg

    original = _job_id()
    task_id = _task(conn, original)
    with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
        conn.execute("UPDATE tasks SET executor_task_id = %s WHERE task_id = %s", (_job_id(), task_id))
    assert conn.execute("SELECT executor_task_id FROM tasks WHERE task_id = %s", (task_id,)).fetchone()[0] == original


def test_clearing_an_id_is_refused(conn):
    import psycopg

    task_id = _task(conn, _job_id())
    with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
        conn.execute("UPDATE tasks SET executor_task_id = NULL WHERE task_id = %s", (task_id,))
