import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ.setdefault("ARTIFACT_ROOT", str(tmp_path_factory.mktemp("downloads")))
    os.environ.setdefault("STATIC_ROOT", "/nonexistent")

    from app.main import app

    # No context manager: the lifespan would open the database pool, and the
    # documentation endpoints are served from source files alone.
    return TestClient(app)


def test_spa_document_is_revalidated_on_every_load(client, tmp_path, monkeypatch):
    import app.main as main

    index = tmp_path / "index.html"
    index.write_text("<!doctype html>", encoding="utf-8")
    monkeypatch.setattr(main, "_INDEX", index)

    response = client.get("/docs")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
