"""Public convergence curves on the leaderboard.

Anyone may see how methods converge on a problem, but only as an aggregate:
median and quartiles per optimizer. A group too small to hide its runs gets no
curve, and nothing in the response leads back to a run or its owner.
"""

import json
import os
import uuid

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not set")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["ARTIFACT_ROOT"] = str(tmp_path_factory.mktemp("downloads"))
    os.environ.setdefault("STATIC_ROOT", "/nonexistent")

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def conn():
    import psycopg

    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        yield connection


def _run(conn, owner, dataset: str, optimizer: str, scale: float) -> str:
    task_id = conn.execute(
        """
        INSERT INTO tasks (queue_name, executor_name, submitted_by, dataset, run_name,
                           model_name, optimizer_name, family, suite, seed, task_status)
        VALUES ('q', 'test', %s, %s, %s, 'mlp-curves', %s, 'gradient', 'test', 1, 'COMPLETED')
        RETURNING task_id
        """,
        (owner, dataset, f"secret-run-{uuid.uuid4().hex[:6]}", optimizer),
    ).fetchone()[0]
    conn.execute(
        """
        INSERT INTO results (task_id, final_loss, final_accuracy, gradient_count,
                             database_reaches, stop_reason)
        VALUES (%s, %s, 0.8, 100, 1000, 'EPOCH_LIMIT')
        """,
        (task_id, 0.5 * scale),
    )
    conn.execute(
        """
        INSERT INTO result_series (task_id, epochs, loss, gradient_count, database_reaches)
        VALUES (%s, '{1,2,3}', %s, '{10,50,100}', '{100,500,1000}')
        """,
        (task_id, [0.9 * scale, 0.7 * scale, 0.5 * scale]),
    )
    return str(task_id)


@pytest.fixture(scope="module")
def problem(conn):
    owner = conn.execute(
        """
        INSERT INTO users (email, auth_provider, role, password_hash)
        VALUES (%s, 'email', 'verified', 'not-a-usable-hash') RETURNING id
        """,
        (f"curves-{uuid.uuid4()}@example.test",),
    ).fetchone()[0]
    dataset = f"curves-{uuid.uuid4().hex[:8]}"
    ids = [_run(conn, owner, dataset, "crowded", s) for s in (1.0, 1.2, 1.4)]
    ids.append(_run(conn, owner, dataset, "lonely", 2.0))
    return {"dataset": dataset, "owner": str(owner), "ids": ids}


def _curves(client, problem, **params):
    response = client.get(
        "/api/leaderboard/curves",
        params={"dataset": problem["dataset"], "model": "mlp-curves", "suite": "test", **params},
    )
    assert response.status_code == 200
    return response.json()


def test_anonymous_visitor_sees_the_median_curve(client, problem):
    body = _curves(client, problem)

    assert body["selected"] == {"dataset": problem["dataset"], "model": "mlp-curves", "suite": "test"}
    assert [s["label"] for s in body["series"]] == ["crowded"]
    crowded = body["series"][0]
    assert crowded["n_runs"] == 3
    assert crowded["median"][-1] == pytest.approx(0.6)


def test_small_group_is_named_but_not_drawn(client, problem):
    body = _curves(client, problem)

    assert body["withheld"] == [{"label": "lonely", "n_runs": 1}]
    assert body["min_runs"] == 3


def test_response_does_not_lead_back_to_runs_or_owners(client, problem):
    text = json.dumps(_curves(client, problem))

    for task_id in problem["ids"]:
        assert task_id not in text
    assert problem["owner"] not in text
    assert "secret-run" not in text


def test_problem_list_offers_only_drawable_problems(client, problem):
    cells = _curves(client, problem)["cells"]
    mine = [c for c in cells if c["dataset"] == problem["dataset"]]

    assert mine == [
        {"dataset": problem["dataset"], "model": "mlp-curves", "suite": "test", "n_optimizers": 1, "n_runs": 4}
    ]


def test_unknown_axis_is_rejected(client, problem):
    response = client.get("/api/leaderboard/curves", params={"x": "seed"})
    assert response.status_code == 400


def test_grid_ends_exactly_on_the_largest_budget():
    from app.services import series

    for logarithmic in (False, True):
        grid = series.build_grid([[(100.0, 1.0), (1000.0, 0.5)]], 7, logarithmic)
        assert grid[0] == 100.0
        assert grid[-1] == 1000.0
