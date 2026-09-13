import json
import os

import pika


def _parameters() -> pika.ConnectionParameters:
    return pika.ConnectionParameters(
        host=os.environ.get("RABBITMQ_HOST", "rabbitmq"),
        port=int(os.environ.get("RABBITMQ_PORT", "5672")),
        credentials=pika.PlainCredentials(os.environ["RABBITMQ_USER"], os.environ["RABBITMQ_PASSWORD"]),
        heartbeat=30,
        blocked_connection_timeout=30,
    )


def publish(exchange: str, routing_key: str, messages: list[dict]) -> None:
    """Publish every message on one short-lived connection. Blocking; raises on failure."""
    connection = pika.BlockingConnection(_parameters())
    try:
        channel = connection.channel()
        channel.confirm_delivery()
        for message in messages:
            channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=json.dumps(message).encode(),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # persist across a broker restart
                    content_type="application/json",
                ),
            )
    finally:
        connection.close()
