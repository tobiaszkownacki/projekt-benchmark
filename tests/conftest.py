import os

from shared.connectors.postgres import PostGresConnector
import pytest
import httpx
from shared.connectors.rabbitmq import RabbitMQConnector


TEST_USER_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def db_connector():

    with PostGresConnector() as db:
        db.conn.autocommit = True
        db.execute("TRUNCATE TABLE users RESTART IDENTITY CASCADE;")
        db.execute(
            "INSERT INTO users (id, email, auth_provider, password_hash) VALUES (%s, %s, 'email', 'x')",
            (TEST_USER_ID, "test@test.pl"),
        )
        yield db


@pytest.fixture
def test_user_id():
    return TEST_USER_ID


@pytest.fixture
def rabbitmq_connector():

    with RabbitMQConnector(exchange="",routing_key="") as publisher:
        queues2clean = ["fake_downloader_queue","fake_worker_queue","dlq_fake_downloader_queue","dlq_fake_worker_queue"]

        for q in queues2clean:
            publisher.channel.queue_purge(queue=q)

        yield publisher

@pytest.fixture
def api_client():
    with httpx.Client(base_url=os.getenv("API_BASE_URL")) as client:
        yield client
