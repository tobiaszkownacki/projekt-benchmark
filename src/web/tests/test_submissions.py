"""A task carries the executor it was queued for.

executor_name was written as the literal "athena" next to a queue name derived
from EXECUTOR, so a deployment on any other executor recorded its tasks under
the wrong name.
"""

import os
import uuid

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not set")


@pytest.fixture
def conn():
    import psycopg

    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        yield connection


@pytest.fixture
def client(tmp_path_factory, conn):
    os.environ["ARTIFACT_ROOT"] = str(tmp_path_factory.mktemp("downloads"))
    os.environ.setdefault("STATIC_ROOT", "/nonexistent")

    from app.main import app
    from app.security import CurrentUser, require_verified

    with TestClient(app) as test_client:
        email = f"submitter-{uuid.uuid4()}@example.test"
        user_id = conn.execute(
            """
            INSERT INTO users (email, auth_provider, role, password_hash)
            VALUES (%s, 'email', 'verified', 'not-a-usable-hash')
            RETURNING id
            """,
            (email,),
        ).fetchone()[0]
        user = CurrentUser(
            id=user_id, email=email, role="verified", display_name=None, is_active=True, has_join_info=True
        )
        app.dependency_overrides[require_verified] = lambda: user
        try:
            yield test_client
        finally:
            app.dependency_overrides.pop(require_verified, None)


def test_a_task_is_recorded_under_the_configured_executor(client, conn):
    from app.settings import settings

    response = client.post(
        "/api/submissions",
        json={
            "display_name": "adam",
            "kind": "builtin",
            "builtin_name": "adam",
            "dataset": "wine_quality",
            "model": "mlp-2x32",
            "max_epochs": 1,
        },
    )

    assert response.status_code == 201, response.text
    task_id = response.json()["task_ids"][0]
    row = conn.execute("SELECT executor_name, queue_name FROM tasks WHERE task_id = %s", (task_id,)).fetchone()
    assert row == (settings.executor_name, settings.worker_queue)
