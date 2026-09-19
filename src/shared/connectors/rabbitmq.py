import json
import os

import pika

from shared.connectors.base import MessageBrokerConnector
from shared.queue_topology import QueueTopology


class RabbitMQConnector(MessageBrokerConnector):
    def __init__(
        self,
        exchange: str,
        routing_key: str,
        exchange_type: str = "direct",
    ) -> None:
        self.exchange = exchange
        self.routing_key = routing_key
        self.exchange_type = exchange_type
        self.user = os.environ.get("RABBITMQ_USER")
        self.password = os.environ.get("RABBITMQ_PASSWORD")
        self.host = os.environ.get("RABBITMQ_HOST", "rabbitmq")
        self.port = os.environ.get("RABBITMQ_PORT", 5672)
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


def declare_topology(channel, topology: QueueTopology) -> None:

    channel.exchange_declare(exchange=topology.main_exchange, exchange_type="direct", durable=True)
    channel.exchange_declare(exchange=topology.dlx_exchange, exchange_type="direct", durable=True)

    for queue, dlq_queue, dlq_routing_key in (
        (topology.worker_queue, topology.worker_dlq_queue, topology.worker_dlq_routing_key),
        (topology.downloader_queue, topology.downloader_dlq_queue, topology.downloader_dlq_routing_key),
    ):
        channel.queue_declare(queue=dlq_queue, durable=True)
        channel.queue_bind(queue=dlq_queue, exchange=topology.dlx_exchange, routing_key=dlq_routing_key)
        channel.queue_declare(
            queue=queue,
            durable=True,
            arguments={
                "x-dead-letter-exchange": topology.dlx_exchange,
                "x-dead-letter-routing-key": dlq_routing_key,
            },
        )
        channel.queue_bind(queue=queue, exchange=topology.main_exchange, routing_key=queue)
