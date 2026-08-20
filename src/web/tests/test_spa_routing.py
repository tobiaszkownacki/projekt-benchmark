"""The SPA catch-all.

§11.2 makes "every resource has its own URL" a requirement rather than a
convenience -- a participant wants to paste a link to their result into an email
or a paper. A route that works only while navigating inside the application does
not satisfy that, and forgetting the server-side catch-all is the standard way
to get exactly that failure. Hence a test.
"""

import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not set")

DEEP_LINKS = [
    "/",
    "/leaderboard",
    "/leaderboard?dataset=wine&family=gradient_free&suite=test&score=loss_v1",
    "/runs",
    "/runs?mine=1",
    "/runs/9f2c4b1e-0000-4000-8000-000000000000",
    "/runs/9f2c4b1e-0000-4000-8000-000000000000/files?path=reports/loss_vs_grads.png",
    "/compare?runs=a,b,c&x=gradient_count&logy=1",
    "/submit",
    "/docs",
    "/admin",
    "/admin/queue",
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    static = tmp_path_factory.mktemp("static")
    (static / "index.html").write_text("<!doctype html><div id=root></div>")
    (static / "assets").mkdir()

    os.environ["STATIC_ROOT"] = str(static)

    # Settings are read once at import, so the module has to load after the
    # environment is in place.
    for module in [m for m in list(__import__("sys").modules) if m.startswith("app")]:
        del __import__("sys").modules[module]

    from app.main import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("path", DEEP_LINKS)
def test_deep_link_serves_the_application(client, path):
    """Opening any of these cold must return the app, not a 404."""
    response = client.get(path)
    assert response.status_code == 200, path
    assert "id=root" in response.text


def test_unknown_api_path_is_still_404(client):
    """The catch-all must not swallow API mistakes into a 200 page."""
    assert client.get("/api/definitely-not-a-route").status_code == 404
