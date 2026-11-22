"""Who may see a run.

A run belongs to the account that submitted it; administrators see every run;
everyone else, signed in or not, is told the run does not exist. The aggregate
views -- the leaderboard and the overview -- still count every run.

Each case is asked as four people: nobody, the owner, another participant and
an administrator.
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


def _account(conn, role: str) -> tuple[uuid.UUID, dict]:
    """A user with a bearer token; returns the id and the request headers."""
    user_id = conn.execute(
        """
        INSERT INTO users (email, auth_provider, role, password_hash)
        VALUES (%s, 'email', %s, 'not-a-usable-hash')
        RETURNING id
        """,
        (f"visibility-{uuid.uuid4()}@example.test", role),
    ).fetchone()[0]
    raw = "bmk_" + secrets.token_urlsafe(32)
    conn.execute(
        """
        INSERT INTO api_tokens (user_id, name, token_sha256, prefix)
        VALUES (%s, 'test', %s, %s)
        """,
        (user_id, hashlib.sha256(raw.encode()).hexdigest(), raw[:10]),
    )
    return user_id, {"Authorization": f"Bearer {raw}"}


def _completed_run(conn, owner: uuid.UUID, optimizer: str) -> str:
    task_id = conn.execute(
        """
        INSERT INTO tasks (queue_name, executor_name, submitted_by, dataset, run_name,
                           model_name, optimizer_name, family, suite, seed, task_status)
        VALUES ('q', 'test', %s, 'wine_quality', %s, 'mlp-1x16', %s, 'gradient',
                'test', 1, 'COMPLETED')
        RETURNING task_id
        """,
        (owner, f"{optimizer}-run", optimizer),
    ).fetchone()[0]
    conn.execute(
        """
        INSERT INTO results (task_id, final_loss, final_accuracy, gradient_count,
                             database_reaches, stop_reason)
        VALUES (%s, 0.5, 0.8, 100, 1000, 'EPOCH_LIMIT')
        """,
        (task_id,),
    )
    conn.execute(
        """
        INSERT INTO result_series (task_id, epochs, loss, gradient_count, database_reaches)
        VALUES (%s, '{1,2,3}', '{0.9,0.7,0.5}', '{10,50,100}', '{100,500,1000}')
        """,
        (task_id,),
    )
    return str(task_id)


@pytest.fixture(scope="module")
def world(conn):
    """Two participants with one run each, plus an administrator.

    The optimizer names are unique per module run, so the leaderboard rows these
    runs form cannot merge with rows other tests created.
    """
    owner_id, owner = _account(conn, "verified")
    other_id, other = _account(conn, "verified")
    _, admin = _account(conn, "admin")
    tag = uuid.uuid4().hex[:8]
    submission = conn.execute(
        """
        INSERT INTO submissions (submitted_by, display_name, kind, source_code, source_sha256, status)
        VALUES (%s, 'private optimizer', 'uploaded', 'def step(): pass', %s, 'accepted')
        RETURNING submission_id
        """,
        (owner_id, "0" * 64),
    ).fetchone()[0]
    return {
        "owner_submission": str(submission),
        "anonymous": {},
        "owner": owner,
        "other": other,
        "admin": admin,
        "owner_run": _completed_run(conn, owner_id, f"own-{tag}"),
        "other_run": _completed_run(conn, other_id, f"other-{tag}"),
        "owner_optimizer": f"own-{tag}",
    }


@pytest.fixture
def artifacts_for(monkeypatch, tmp_path):
    """Point the artifact root at a temporary directory holding one file per run."""
    from app.services import artifacts

    monkeypatch.setattr(artifacts, "artifact_root", lambda: tmp_path.resolve())

    def create(task_id: str) -> None:
        directory = tmp_path / task_id / "reports"
        directory.mkdir(parents=True)
        (directory / "summary.txt").write_text("private result\n")

    return create


def _listed_ids(client, headers, **params) -> set[str]:
    body = client.get("/api/runs", params={"limit": 200, **params}, headers=headers).json()
    return {run["task_id"] for run in body["runs"]}


# --- listing -----------------------------------------------------------------


def test_anonymous_listing_is_refused(client, world):
    assert client.get("/api/runs").status_code == 401
    assert client.get("/api/runs?mine=true").status_code == 401
    assert client.get("/api/runs/filters").status_code == 401


def test_participant_lists_only_their_own_runs(client, world):
    listed = _listed_ids(client, world["owner"])
    assert world["owner_run"] in listed
    assert world["other_run"] not in listed

    body = client.get("/api/runs", headers=world["owner"]).json()
    # The total is computed over the same restriction, not over every run.
    assert body["total"] == len(listed)


def test_participant_search_cannot_reach_another_run(client, world):
    listed = _listed_ids(client, world["owner"], search=world["other_run"])
    assert listed == set()


def test_administrator_lists_every_run(client, world):
    listed = _listed_ids(client, world["admin"])
    assert {world["owner_run"], world["other_run"]} <= listed


def test_administrator_mine_lists_only_their_own(client, world):
    listed = _listed_ids(client, world["admin"], mine="true")
    assert world["owner_run"] not in listed
    assert world["other_run"] not in listed


def test_filter_options_follow_the_listing(client, world):
    own = client.get("/api/runs/filters", headers=world["owner"]).json()
    assert world["owner_optimizer"] in own["optimizers"]
    assert f"other-{world['owner_optimizer'][4:]}" not in own["optimizers"]

    everything = client.get("/api/runs/filters", headers=world["admin"]).json()
    assert f"other-{world['owner_optimizer'][4:]}" in everything["optimizers"]


# --- run-scoped endpoints ------------------------------------------------------

RUN_SCOPED = (
    "",
    "/transitions",
    "/series",
    "/files",
    "/files/raw?path=reports/summary.txt",
    "/files/meta?path=reports/summary.txt",
    "/archive.zip",
)


@pytest.mark.parametrize("suffix", RUN_SCOPED)
@pytest.mark.parametrize("who", ["anonymous", "other"])
def test_run_is_not_found_for_anyone_but_owner_and_admin(client, world, artifacts_for, suffix, who):
    artifacts_for(world["owner_run"])
    response = client.get(f"/api/runs/{world['owner_run']}{suffix}", headers=world[who])
    # 404, not 403: a refusal must read the same as a run that does not exist.
    assert response.status_code == 404
    unknown = client.get(f"/api/runs/{uuid.uuid4()}{suffix}", headers=world[who])
    assert unknown.status_code == 404
    assert response.json() == unknown.json()


@pytest.mark.parametrize("suffix", RUN_SCOPED)
@pytest.mark.parametrize("who", ["owner", "admin"])
def test_run_opens_for_owner_and_admin(client, world, artifacts_for, suffix, who):
    artifacts_for(world["owner_run"])
    response = client.get(f"/api/runs/{world['owner_run']}{suffix}", headers=world[who])
    assert response.status_code == 200


def test_archive_carries_the_files_for_the_owner(client, world, artifacts_for):
    import io
    import zipfile

    artifacts_for(world["owner_run"])
    response = client.get(f"/api/runs/{world['owner_run']}/archive.zip", headers=world["owner"])
    assert response.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
    assert names == ["reports/summary.txt"]


def test_detail_says_who_may_manage_it(client, world):
    for who, expected in (("owner", True), ("admin", True)):
        body = client.get(f"/api/runs/{world['owner_run']}", headers=world[who]).json()
        assert body["can_manage"] is expected


@pytest.mark.parametrize(("who", "expected"), [("anonymous", 404), ("other", 404), ("owner", 200), ("admin", 200)])
def test_submission_source_follows_the_same_rule(client, world, who, expected):
    response = client.get(f"/api/submissions/{world['owner_submission']}", headers=world[who])
    assert response.status_code == expected


# --- compare -------------------------------------------------------------------


def _compare(client, headers, path="/api/compare", runs=()):
    return client.get(path, params={"runs": ",".join(runs)}, headers=headers)


@pytest.mark.parametrize("path", ["/api/compare", "/api/compare/export.csv"])
def test_anonymous_compare_is_refused(client, world, path):
    assert _compare(client, {}, path, [world["owner_run"]]).status_code == 401


def test_compare_drops_another_users_run_like_an_unknown_one(client, world):
    forbidden = _compare(client, world["owner"], runs=[world["owner_run"], world["other_run"]]).json()
    unknown = _compare(client, world["owner"], runs=[world["owner_run"], str(uuid.uuid4())]).json()

    assert [r["task_id"] for r in forbidden["runs"]] == [world["owner_run"]]
    assert forbidden["missing"] == 1
    assert forbidden["series"] == unknown["series"]
    assert forbidden["missing"] == unknown["missing"]


def test_compare_shows_every_run_to_an_administrator(client, world):
    body = _compare(client, world["admin"], runs=[world["owner_run"], world["other_run"]]).json()
    assert {r["task_id"] for r in body["runs"]} == {world["owner_run"], world["other_run"]}
    assert body["missing"] == 0


def test_export_contains_only_readable_runs(client, world):
    other_only = _compare(client, world["owner"], "/api/compare/export.csv", [world["other_run"]])
    assert other_only.status_code == 200
    # Header only: the other participant's curve is not exported.
    assert other_only.text.strip().count("\n") == 0

    own = _compare(client, world["owner"], "/api/compare/export.csv", [world["owner_run"]])
    assert world["owner_optimizer"] in own.text

    admin = _compare(client, world["admin"], "/api/compare/export.csv", [world["other_run"]])
    assert f"other-{world['owner_optimizer'][4:]}" in admin.text


# --- aggregates ------------------------------------------------------------------


def _row(client, headers, optimizer):
    rows = client.get("/api/leaderboard", headers=headers).json()["rows"]
    return next(r for r in rows if r["optimizer"] == optimizer)


def test_leaderboard_stays_public_and_counts_every_run(client, world):
    response = client.get("/api/leaderboard")
    assert response.status_code == 200
    optimizers = {r["optimizer"] for r in response.json()["rows"]}
    assert world["owner_optimizer"] in optimizers
    assert f"other-{world['owner_optimizer'][4:]}" in optimizers


@pytest.mark.parametrize(
    ("who", "expected"),
    [("anonymous", False), ("owner", True), ("other", False), ("admin", True)],
)
def test_leaderboard_row_says_whether_the_requester_can_open_it(client, world, who, expected):
    row = _row(client, world[who], world["owner_optimizer"])
    assert row["can_view"] is expected
    assert row["n_runs"] == 1
    # Identifiers are handed out only alongside the permission to open them.
    assert row["task_ids"] == ([world["owner_run"]] if expected else [])


def test_every_leaderboard_row_carries_can_view(client, world):
    for row in client.get("/api/leaderboard").json()["rows"]:
        assert row["can_view"] is False


def test_overview_still_counts_runs_it_does_not_show(client, world):
    body = client.get("/api/overview").json()
    assert body["runs"] >= 2
    assert body["completed"] >= 2


# --- live events -------------------------------------------------------------------


def test_events_reach_only_the_owner_and_administrators():
    from app.routers.events import visible_event
    from app.security import CurrentUser

    def user(role: str, user_id: uuid.UUID | None = None) -> CurrentUser:
        return CurrentUser(
            id=user_id or uuid.uuid4(),
            email="x@example.test",
            role=role,
            display_name=None,
            is_active=True,
            has_join_info=False,
        )

    owner_id = uuid.uuid4()
    # The trigger serialises the owner as text; the comparison must not.
    payload = {"task_id": str(uuid.uuid4()), "status": "RUNNING", "submitted_by": str(owner_id)}

    assert visible_event(payload, user("verified", owner_id)) is True
    assert visible_event(payload, user("admin")) is True
    assert visible_event(payload, user("verified")) is False
    assert visible_event(payload, None) is False

    ownerless = {"task_id": str(uuid.uuid4()), "status": "RUNNING"}
    assert visible_event(ownerless, user("verified")) is False
    assert visible_event(ownerless, None) is False
    assert visible_event(ownerless, user("admin")) is True
