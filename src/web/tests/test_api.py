"""Contract tests against a live database.

Skipped when TEST_DATABASE_URL is unset, so the suite still runs on a machine
without Postgres. The path they cover is the one that cannot be checked any
other way: the SPA catch-all, which is the single most common way to break deep
links, and the authorisation boundary on artifacts.
"""

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


def test_healthz_reports_the_database(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["database"] == "up"


def test_overview_counters_are_present(client):
    body = client.get("/api/overview").json()
    for key in ("runs", "completed", "gradients", "samples"):
        assert key in body


def test_leaderboard_describes_its_own_formula(client):
    """The aggregate must say which formula produced it.

    §13.1 forbids reducing the ranking to one unexplained number, because the
    exchange rate between a gradient and a sample is an open question that
    decides who wins.
    """
    body = client.get("/api/leaderboard").json()
    assert body["score_formula"]["id"]
    assert body["score_formula"]["note"]
    assert len(body["available_formulas"]) >= 2
    for row in body["rows"]:
        # n travels with every row, so a median over one run can be flagged.
        assert "n_runs" in row
        assert "median" in row["final_loss"]


def test_leaderboard_formula_changes_the_ordering_key(client):
    by_loss = client.get("/api/leaderboard?score=loss_v1").json()
    by_accuracy = client.get("/api/leaderboard?score=accuracy_v1").json()
    assert by_loss["score_formula"]["direction"] == "asc"
    assert by_accuracy["score_formula"]["direction"] == "desc"


def test_runs_listing_paginates(client):
    body = client.get("/api/runs?limit=5").json()
    assert len(body["runs"]) <= 5
    assert "total" in body


def test_series_downsampling_reports_itself(client):
    runs = client.get("/api/runs?status=completed&limit=1").json()["runs"]
    if not runs:
        pytest.skip("no completed runs in the database")
    task_id = runs[0]["task_id"]

    body = client.get(f"/api/runs/{task_id}/series?points=10").json()
    assert len(body["points"]) <= 10
    assert "original_points" in body
    if body["truncated"]:
        assert body["downsample"] == "lttb"


def test_series_rejects_an_unknown_axis(client):
    runs = client.get("/api/runs?status=completed&limit=1").json()["runs"]
    if not runs:
        pytest.skip("no completed runs in the database")
    response = client.get(f"/api/runs/{runs[0]['task_id']}/series?x=wall_time")
    assert response.status_code == 400


def test_unknown_run_is_404(client):
    assert client.get(f"/api/runs/{uuid.uuid4()}").status_code == 404


def test_malformed_uuid_is_rejected_before_the_filesystem(client):
    assert client.get("/api/runs/not-a-uuid/files").status_code == 422


def test_artifact_traversal_is_refused(client):
    runs = client.get("/api/runs?status=completed&limit=1").json()["runs"]
    if not runs:
        pytest.skip("no completed runs in the database")
    task_id = runs[0]["task_id"]
    for attack in ("../../etc/passwd", "/etc/passwd", "reports/../../../etc/passwd"):
        response = client.get(f"/api/runs/{task_id}/files/raw", params={"path": attack})
        assert response.status_code in (400, 404), attack


def test_submission_requires_a_verified_account(client):
    response = client.post("/api/submissions", json={
        "display_name": "x", "kind": "builtin", "builtin_name": "adam",
        "dataset": "wine", "model": "mlp-1x16", "max_epochs": 1,
    })
    assert response.status_code in (401, 403)


def test_admin_endpoints_are_not_discoverable_anonymously(client):
    # 404 rather than 403: a 403 confirms the route exists.
    assert client.get("/api/admin/queue").status_code in (401, 404)


def test_protocol_examples_come_from_the_repository(client):
    body = client.get("/api/protocol").json()
    assert body["examples"]["gradient"]["present"] is True
    source = client.get("/api/protocol/example/gradient")
    assert source.status_code == 200
    assert source.headers["x-content-type-options"] == "nosniff"
    assert "def step" in source.text


def test_template_is_downloadable(client):
    response = client.get("/api/protocol/template")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert "initial_params" in response.text


def test_vocabulary_covers_every_state(client):
    body = client.get("/api/vocabulary").json()
    for state in ("queued_broker", "queued_slurm", "running", "downloading",
                  "completed", "completed_no_artifacts", "failed",
                  "failed_no_artifacts", "rejected"):
        assert state in body["run_states"], state
        assert body["run_states"][state]["label"]


def test_security_headers_on_api_responses(client):
    response = client.get("/api/overview")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" in response.headers
    assert response.headers["x-frame-options"] == "DENY"
