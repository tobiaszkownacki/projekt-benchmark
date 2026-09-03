import json
import logging
from dataclasses import asdict
from typing import Any

from core.config import get_rabbitmq_connection_params

logger = logging.getLogger(__name__)


class RabbitMQConnector:
    """Safe RabbitMQ publisher for Streamlit frontend with fallback when RMQ is offline."""

    def __init__(
        self,
        exchange: str = "main-exchange",
        routing_key: str = "ATHENA_WORKER_QUEUE",
        exchange_type: str = "direct",
        user: str | None = None,
        password: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self.exchange = exchange
        self.routing_key = routing_key
        self.exchange_type = exchange_type
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.connection = None
        self.channel = None
        self.is_connected = False

    def __enter__(self) -> "RabbitMQConnector":
        import pika

        try:
            if self.user and self.password and self.host and self.port:
                credentials = pika.PlainCredentials(self.user, self.password)
                params = pika.ConnectionParameters(
                    host=self.host,
                    port=self.port,
                    credentials=credentials,
                    connection_attempts=1,
                    retry_delay=1,
                    socket_timeout=2.0,
                )
            else:
                params = get_rabbitmq_connection_params()
                params.connection_attempts = 1
                params.socket_timeout = 2.0

            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()
            self.channel.exchange_declare(
                exchange=self.exchange,
                exchange_type=self.exchange_type,
                durable=True,
            )
            self.is_connected = True
        except Exception as exc:
            logger.info("RabbitMQ is not reachable (%s); task queued in simulation mode.", exc)
            self.is_connected = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.connection and not self.connection.is_closed:
            try:
                self.connection.close()
            except Exception:
                pass

    def publish(self, payload: dict | Any) -> bool:
        if hasattr(payload, "__dataclass_fields__"):
            payload_dict = asdict(payload)
        elif isinstance(payload, dict):
            payload_dict = payload
        else:
            payload_dict = {"data": str(payload)}

        if not self.is_connected or not self.channel:
            logger.info("Simulation mode: message dispatched for %s", payload_dict.get("task_id"))
            return False

        import pika

        self.channel.basic_publish(
            exchange=self.exchange,
            routing_key=self.routing_key,
            body=json.dumps(payload_dict),
            properties=pika.BasicProperties(
                delivery_mode=pika.DeliveryMode.Persistent,
                content_type="application/json",
            ),
        )
        return True
