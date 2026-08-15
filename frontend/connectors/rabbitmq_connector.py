import json
import os

import pika


class RabbitMQConnector:

    def __init__(
        self,
        exchange: str,
        routing_key: str,
        exchange_type: str = "direct",
        user: str | None = None,
        password: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self.exchange = exchange
        self.routing_key = routing_key
        self.exchange_type = exchange_type
        self.user = user or os.environ.get("RABBITMQ_USER")
        self.password = password or os.environ.get("RABBITMQ_PASSWORD")
        self.host = host or os.environ.get("RABBITMQ_HOST", "rabbitmq")
        self.port = port or os.environ.get("RABBITMQ_PORT", 5672)
        self.connection = None
        self.channel = None

    def __enter__(self) -> "RabbitMQConnector":
        credentials = pika.PlainCredentials(self.user, self.password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=self.host, port=self.port, credentials=credentials)
        )
        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange=self.exchange, exchange_type=self.exchange_type, durable=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.connection and not self.connection.is_closed:
            self.connection.close()

    def publish(self, payload: dict) -> None:
        self.channel.basic_publish(
            exchange=self.exchange,
            routing_key=self.routing_key,
            body=json.dumps(payload),
            properties=pika.BasicProperties(
                delivery_mode=pika.DeliveryMode.Persistent,
                content_type="application/json",
            ),
        )
