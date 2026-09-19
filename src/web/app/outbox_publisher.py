"""Drains queue_outbox into RabbitMQ.

A separate process on purpose: pika blocks and is not async-safe, so running
it outside any event loop avoids working around that with a thread pool. It
also keeps broker credentials out of the API process entirely.

Delivery is at-least-once. A message published just before the row is marked
published gets sent twice after a crash, which is why the worker is expected
to treat task_id as an idempotency key.

Run with:  python -m app.outbox_publisher
"""

import json
import logging
import os
import pathlib
import signal
import sys
import time

import pika
import psycopg
from psycopg.rows import dict_row

from app.services.outbox import MAX_ATTEMPTS
from shared.connectors.rabbitmq import declare_topology
from shared.queue_topology import QueueTopology

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s outbox: %(message)s",
)
logger = logging.getLogger(__name__)

# Touched on every pass of the drain loop and checked by the container
# healthcheck. A liveness probe that only asks whether the process exists would
# report a wedged loop as healthy, which is the failure worth catching.
HEARTBEAT = pathlib.Path(os.environ.get("OUTBOX_HEARTBEAT", "/tmp/outbox-heartbeat"))

POLL_SECONDS = float(os.environ.get("OUTBOX_POLL_SECONDS", "1.0"))
BATCH = int(os.environ.get("OUTBOX_BATCH", "50"))

# Publishing to an exchange nobody declared closes the channel, and publishing
# to one with no matching binding drops the message while the row is marked
# published. Both are answered by declaring the topology here as well: the
# drain process is the first thing to touch the broker on a cold start.
_PERMANENT = (pika.exceptions.UnroutableError, pika.exceptions.ChannelClosedByBroker)

_running = True


def _stop(*_args):
    global _running
    _running = False
    logger.info("Shutdown requested")


def connection_parameters() -> pika.ConnectionParameters:
    return pika.ConnectionParameters(
        host=os.environ.get("RABBITMQ_HOST", "rabbitmq"),
        port=int(os.environ.get("RABBITMQ_PORT", "5672")),
        credentials=pika.PlainCredentials(os.environ["RABBITMQ_USER"], os.environ["RABBITMQ_PASSWORD"]),
        heartbeat=30,
        blocked_connection_timeout=30,
    )


def publish_batch(db: psycopg.Connection, channel) -> int:
    """Publish up to BATCH pending rows. Returns how many were sent."""
    with db.cursor(row_factory=dict_row) as cursor:
        # FOR UPDATE SKIP LOCKED so more than one publisher is safe.
        cursor.execute(
            """
            SELECT id, exchange, routing_key, payload, attempts
              FROM queue_outbox
             WHERE published_at IS NULL AND attempts < %s
             ORDER BY id
             LIMIT %s
             FOR UPDATE SKIP LOCKED
            """,
            (MAX_ATTEMPTS, BATCH),
        )
        rows = cursor.fetchall()

        sent = 0
        for row in rows:
            body = row["payload"]
            if not isinstance(body, str):
                body = json.dumps(body)
            try:
                channel.basic_publish(
                    exchange=row["exchange"],
                    routing_key=row["routing_key"],
                    body=body.encode(),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # persist across broker restart
                        content_type="application/json",
                        message_id=str(row["id"]),
                    ),
                    # Without this the broker discards a message that matches no
                    # binding and says nothing, and the row is marked published.
                    mandatory=True,
                )
                cursor.execute(
                    "UPDATE queue_outbox SET published_at = NOW() WHERE id = %s",
                    (row["id"],),
                )
                sent += 1
            except _PERMANENT as exc:
                # The message or the topology is wrong, and trying it forever
                # would drain the whole batch behind it. This is what the
                # attempt budget is for.
                logger.warning("Publish of row %s failed: %s", row["id"], exc)
                cursor.execute(
                    """
                    UPDATE queue_outbox
                       SET attempts = attempts + 1, last_error = %s
                     WHERE id = %s
                    """,
                    (f"{type(exc).__name__}: {exc}"[:500], row["id"]),
                )
                db.commit()
                raise
            except Exception as exc:
                # The broker went away. Charging the budget for that abandons
                # submissions after a minute of downtime, so the row keeps its
                # attempts and the marks earned earlier in the batch are kept
                # rather than rolled back into a second delivery.
                logger.warning("Broker unavailable while publishing row %s: %s", row["id"], exc)
                db.commit()
                raise
    db.commit()
    return sent


def main() -> int:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL is not set")
        return 1

    executor = os.environ.get("EXECUTOR")
    if not executor:
        logger.error("EXECUTOR is not set, so the queues to declare are unknown")
        return 1
    topology = QueueTopology(executor)

    while _running:
        db = None
        broker = None
        try:
            db = psycopg.connect(database_url, autocommit=False)
            broker = pika.BlockingConnection(connection_parameters())
            channel = broker.channel()
            channel.confirm_delivery()
            # Re-declared on every connection: the publisher may be the first
            # process to reach a freshly started broker, and a message sent
            # before the pipeline declares its queues has nowhere to go.
            declare_topology(channel, topology)
            logger.info("Connected; draining %s every %.1fs", topology.worker_queue, POLL_SECONDS)

            while _running:
                try:
                    HEARTBEAT.touch()
                except OSError:
                    pass
                sent = publish_batch(db, channel)
                if sent:
                    logger.info("Published %d message(s)", sent)
                broker.process_data_events(time_limit=0)
                time.sleep(POLL_SECONDS if not sent else 0.05)
        except Exception as exc:
            logger.warning("Publisher loop failed (%s); retrying in 5s", exc)
            time.sleep(5)
        finally:
            for resource in (broker, db):
                try:
                    if resource is not None:
                        resource.close()
                except Exception:
                    pass

    logger.info("Stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
