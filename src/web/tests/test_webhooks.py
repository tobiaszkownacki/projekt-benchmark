"""The callback endpoint is public, so its refusals matter as much as its work.

Every case here is one the cluster actually produces: a trap that fires twice
on the way out, a retry from curl, a job id that belongs to a different task,
and a token that outlived its run.
"""

import hashlib
import os
import secrets
import uuid

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not set")

CALLBACK = "/api/executors/callback"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["ARTIFACT_ROOT"] = str(tmp_path_factory.mktemp("downloads"))
    os.environ.setdefault("STATIC_ROOT", "/nonexistent")

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def conn():
    import psycopg

    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        yield connection


def _task(conn, *, submitted=True, ttl_hours=48) -> tuple[str, str, str]:
    """A task with a webhook token, as a submission would have created it.

    The cluster job id is unique per task: the schema makes it write-once, so a
    literal shared by two tests would fail the second insert.
    """
    executor_task_id = f"job-{uuid.uuid4().hex[:12]}" if submitted else None
    email = f"webhook-{uuid.uuid4()}@example.test"
    user_id = conn.execute(
        """
        INSERT INTO users (email, auth_provider, role, password_hash)
        VALUES (%s, 'email', 'verified', 'not-a-usable-hash')
        RETURNING id
        """,
        (email,),
    ).fetchone()[0]
    task_id = conn.execute(
        """
        INSERT INTO tasks (queue_name, executor_name, submitted_by, dataset, run_name, executor_task_id)
        VALUES ('q', 'test', %s, 'wine_quality', 'run', %s)
        RETURNING task_id
        """,
        (user_id, executor_task_id),
    ).fetchone()[0]

    raw = "bmw_" + secrets.token_urlsafe(32)
    conn.execute(
        """
        INSERT INTO task_webhook_tokens (task_id, token_sha256, prefix, expires_at)
        VALUES (%s, %s, %s, NOW() + make_interval(hours => %s))
        """,
        (task_id, hashlib.sha256(raw.encode()).hexdigest(), raw[:10], ttl_hours),
    )
    return str(task_id), raw, executor_task_id


def _status(conn, task_id: str) -> tuple[str, int | None]:
    row = conn.execute("SELECT task_status::text, exit_code FROM tasks WHERE task_id = %s", (task_id,)).fetchone()
    return row[0], row[1]


def test_a_callback_without_a_token_is_refused(client):
    response = client.post(f"{CALLBACK}?job_id=1&state=COMPLETED&exit_code=0")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_an_unknown_token_is_refused(client):
    response = client.post(
        f"{CALLBACK}?job_id=1&state=COMPLETED&exit_code=0",
        headers={"Authorization": "Bearer bmw_nothing"},
    )

    assert response.status_code == 401


def test_an_expired_token_is_refused(client, conn):
    _, token, job_id = _task(conn, ttl_hours=-1)

    response = client.post(
        f"{CALLBACK}?job_id={job_id}&state=COMPLETED&exit_code=0",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


def test_an_unknown_state_is_refused(client, conn):
    _, token, job_id = _task(conn)

    response = client.post(
        f"{CALLBACK}?job_id={job_id}&state=TIMEOUT&exit_code=0",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 422


def test_a_completed_job_closes_its_task_and_asks_for_the_artifacts(client, conn):
    task_id, token, job_id = _task(conn)

    body = client.post(
        f"{CALLBACK}?job_id={job_id}&state=COMPLETED&exit_code=0",
        headers={"Authorization": f"Bearer {token}"},
    ).json()

    assert body == {"task_id": task_id, "status": "COMPLETED", "applied": True}
    assert _status(conn, task_id) == ("COMPLETED", 0)
    queued = conn.execute("SELECT COUNT(*) FROM queue_outbox WHERE payload->>'task_id' = %s", (task_id,)).fetchone()[0]
    assert queued == 1


def test_a_zero_exit_that_reports_failure_is_a_failure(client, conn):
    task_id, token, job_id = _task(conn)

    client.post(
        f"{CALLBACK}?job_id={job_id}&state=FAILED&exit_code=137",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert _status(conn, task_id) == ("FAILED", 137)


def test_the_second_hit_of_the_same_trap_changes_nothing(client, conn):
    task_id, token, job_id = _task(conn)
    url = f"{CALLBACK}?job_id={job_id}&state=COMPLETED&exit_code=0"
    headers = {"Authorization": f"Bearer {token}"}

    first = client.post(url, headers=headers).json()
    second = client.post(url, headers=headers).json()

    assert first["applied"] is True
    assert second["applied"] is False
    assert second["task_id"] == task_id


def test_a_job_id_from_another_task_is_a_conflict(client, conn):
    task_id, token, _ = _task(conn)

    response = client.post(
        f"{CALLBACK}?job_id=job-somebody-elses&state=COMPLETED&exit_code=0",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 409
    assert _status(conn, task_id)[0] == "PENDING"


def test_a_callback_that_beat_the_worker_records_the_job_id(client, conn):
    task_id, token, _ = _task(conn, submitted=False)
    job_id = f"job-early-{uuid.uuid4().hex[:12]}"

    client.post(
        f"{CALLBACK}?job_id={job_id}&state=COMPLETED&exit_code=0",
        headers={"Authorization": f"Bearer {token}"},
    )

    row = conn.execute("SELECT executor_task_id FROM tasks WHERE task_id = %s", (task_id,)).fetchone()
    assert row[0] == job_id
    assert _status(conn, task_id) == ("COMPLETED", 0)


def test_a_spent_token_stops_working(client, conn):
    task_id, token, job_id = _task(conn)
    conn.execute("UPDATE task_webhook_tokens SET uses = 10 WHERE task_id = %s", (task_id,))

    response = client.post(
        f"{CALLBACK}?job_id={job_id}&state=COMPLETED&exit_code=0",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
