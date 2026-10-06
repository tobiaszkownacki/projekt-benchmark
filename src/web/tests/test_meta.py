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


def test_api_carries_the_official_name(client):
    from app.main import app

    assert app.title == "Opti-Mark (The Golem Benchmark)"


def test_protocol_no_longer_lists_contract_limitations(client):
    body = client.get("/api/protocol").json()

    assert "known_limitations" not in body
    assert set(body) == {"evaluator_api", "examples", "sandbox", "stop_reasons"}


def test_protocol_serves_every_example(client):
    examples = client.get("/api/protocol").json()["examples"]

    assert set(examples) == {"gradient", "evolutionary", "protocol", "base"}
    for name, example in examples.items():
        assert example["present"] is True, name
        response = client.get(f"/api/protocol/example/{name}")
        assert response.status_code == 200, name
        assert response.text.strip(), name


def test_template_names_the_benchmark(client):
    response = client.get("/api/protocol/template")

    assert response.status_code == 200
    assert "Optimizer template for Opti-Mark (The Golem Benchmark)." in response.text
    assert "Black-Box Benchmark" not in response.text


def test_spa_document_is_revalidated_on_every_load(client, tmp_path, monkeypatch):
    import app.main as main

    index = tmp_path / "index.html"
    index.write_text("<!doctype html>", encoding="utf-8")
    monkeypatch.setattr(main, "_INDEX", index)

    response = client.get("/docs")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
