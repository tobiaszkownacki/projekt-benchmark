"""Queue publication through a transactional outbox.

Writing the task and publishing the message are two operations with no shared
transaction: lose the broker between them and the run exists but never
starts; lose the process after publishing but before committing and a job
runs that no row describes.

Inserting the message into queue_outbox inside the same transaction as the
task makes the pair atomic. A separate drain process publishes with ordinary
blocking pika, outside any event loop, so the API keeps no broker credentials
at all.

The cost is a publication delay of about a second, which is negligible given
typical submission rates.
"""

import json
import os
from typing import Any
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from app.settings import settings

# Read by the drain process too, so the budget a row is measured against and
# the budget the interface reports it against cannot drift apart.
MAX_ATTEMPTS = int(os.environ.get("OUTBOX_MAX_ATTEMPTS", "10"))

# Bumped when a field is removed, retyped or given a new meaning. The consumer
# reads exactly one version, so a message written before a deployment and
# published after it is refused rather than misread.
SCHEMA_VERSION = 1


async def enqueue(
    conn: psycopg.AsyncConnection,
    payload: dict[str, Any],
    routing_key: str | None = None,
    exchange: str | None = None,
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


def task_message(
    task_id: UUID,
    *,
    run_name: str,
    dataset: str,
    model: str,
    optimizers: list[str],
    seed: int,
    stop_condition: dict[str, int],
    webhook_token: str | None = None,
) -> dict[str, Any]:
    """The whole run, as pipeline.executor.JobDescription.from_message reads it.

    A compute node cannot reach the database, so whatever is missing here is
    missing for good: the seed, the model and the stop condition used to stay
    behind in the row and the cluster ran defaults instead.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": str(task_id),
        "run_name": run_name,
        "dataset": dataset,
        "model": model,
        "optimizers": list(optimizers),
        "seed": int(seed),
        "stop_condition": {key: int(value) for key, value in stop_condition.items()},
        "webhook_token": webhook_token,
    }


async def pending_count() -> int:
    """Messages still waiting for a publisher that will still try them."""
    from app import db

    row = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM queue_outbox WHERE published_at IS NULL AND attempts < %s",
        (MAX_ATTEMPTS,),
    )
    return int(row["n"]) if row else 0


async def abandoned_count() -> int:
    """Messages nobody will publish again.

    Counted apart from the pending ones because they are not slow, they are
    lost: the row describes a submission that exists in tasks and will never
    reach the cluster, and nothing else in the system says so.
    """
    from app import db

    row = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM queue_outbox WHERE published_at IS NULL AND attempts >= %s",
        (MAX_ATTEMPTS,),
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
