"""Queue publication through a transactional outbox.

§15 warns that pika blocks and is not async-safe, and recommends aio-pika. That
solves half the problem. The other half is that writing the task and publishing
the message are two operations with no shared transaction: lose the broker
between them and the run exists but never starts; lose the process after
publishing but before committing and a job runs that no row describes.

Inserting the message into queue_outbox inside the same transaction as the task
makes the pair atomic. A separate drain process publishes with ordinary blocking
pika, outside any event loop -- so the async-safety question disappears rather
than being worked around, and the API keeps no broker credentials at all, which
was the point of §9's complaint about the UI holding infrastructure secrets.

The cost is a publication delay of about a second. At one to three submissions
per user per day (§5.2) that is not a cost.
"""

import json
from typing import Any, Optional
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from app.settings import settings


async def enqueue(
    conn: psycopg.AsyncConnection,
    payload: dict[str, Any],
    routing_key: Optional[str] = None,
    exchange: Optional[str] = None,
) -> None:
    """Add a message to the outbox inside the caller's transaction."""
    await conn.execute(
        """
        INSERT INTO queue_outbox (exchange, routing_key, payload)
        VALUES (%s, %s, %s)
        """,
        (
            exchange if exchange is not None else settings.main_exchange,
            routing_key or settings.worker_queue,
            Jsonb(payload),
        ),
    )


def task_message(task_id: UUID, queue_name: str) -> dict[str, Any]:
    """The message shape the existing Athena worker already consumes."""
    return {"task_id": str(task_id), "queue_name": queue_name}


async def pending_count() -> int:
    from app import db

    row = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM queue_outbox WHERE published_at IS NULL"
    )
    return int(row["n"]) if row else 0


async def recent(limit: int = 50) -> list[dict]:
    from app import db

    rows = await db.fetch_all(
        """
        SELECT id, exchange, routing_key, payload, created_at,
               published_at, attempts, last_error
          FROM queue_outbox
         ORDER BY id DESC
         LIMIT %s
        """,
        (limit,),
    )
    for row in rows:
        if isinstance(row.get("payload"), str):
            row["payload"] = json.loads(row["payload"])
    return rows
