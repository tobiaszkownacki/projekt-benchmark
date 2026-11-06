"""A row the broker will never route must not hold up the rows behind it.

The drain loop raised on the first permanent failure, and the next pass began
again from the same row, so every submission queued after it waited until that
row had spent its whole attempt budget.
"""

import asyncio
import json
import os
import uuid

import pytest

pika = pytest.importorskip("pika")

DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not set")


class _Channel:
    """Answers like a confirming channel: an unroutable message comes back and the channel stays open.

    A broker refusal, as for an exchange that does not exist, closes it instead.
    """

    def __init__(self, refused: str, closes: bool = False) -> None:
        self.refused = refused
        self.closes = closes
        self.published: list[str] = []
        self.is_closed = False

    def basic_publish(self, exchange, routing_key, body, properties, mandatory):
        if routing_key == self.refused:
            if self.closes:
                self.is_closed = True
                raise pika.exceptions.ChannelClosedByBroker(404, "NOT_FOUND")
            raise pika.exceptions.UnroutableError([])
        self.published.append(json.loads(body).get("marker"))


@pytest.fixture
def db():
    import psycopg
    from app.migrations import apply_migrations

    asyncio.run(apply_migrations(DATABASE_URL))
    with psycopg.connect(DATABASE_URL, autocommit=False) as connection:
        yield connection


def _row(db, routing_key: str, marker: str) -> int:
    row_id = db.execute(
        "INSERT INTO queue_outbox (exchange, routing_key, payload) VALUES ('main', %s, %s) RETURNING id",
        (routing_key, json.dumps({"marker": marker})),
    ).fetchone()[0]
    db.commit()
    return row_id


def _state(db, row_id: int) -> tuple[int, bool]:
    attempts, published_at = db.execute(
        "SELECT attempts, published_at FROM queue_outbox WHERE id = %s", (row_id,)
    ).fetchone()
    return attempts, published_at is not None


@pytest.fixture
def publisher(monkeypatch):
    from app import outbox_publisher

    # Rows left by other tests in the same database come first by id.
    monkeypatch.setattr(outbox_publisher, "BATCH", 10_000)
    return outbox_publisher


def test_an_unroutable_row_does_not_hold_back_the_row_behind_it(db, publisher):
    nowhere, marker = f"nowhere-{uuid.uuid4().hex}", f"good-{uuid.uuid4().hex}"
    bad = _row(db, nowhere, "bad")
    good = _row(db, "worker", marker)
    channel = _Channel(refused=nowhere)

    publisher.publish_batch(db, channel)

    assert marker in channel.published
    assert _state(db, bad) == (1, False)
    assert _state(db, good) == (0, True)


def test_a_refusal_that_closes_the_channel_ends_the_batch(db, publisher):
    nowhere = f"nowhere-{uuid.uuid4().hex}"
    bad = _row(db, nowhere, "bad")
    good = _row(db, "worker", f"good-{uuid.uuid4().hex}")

    with pytest.raises(pika.exceptions.ChannelClosedByBroker):
        publisher.publish_batch(db, _Channel(refused=nowhere, closes=True))

    assert _state(db, bad) == (1, False)
    assert _state(db, good) == (0, False)
