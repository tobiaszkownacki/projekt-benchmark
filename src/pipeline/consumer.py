import json
import logging
from collections.abc import Callable

from shared.connectors.base import MessageBrokerConnector
from shared.connectors.rabbitmq import RabbitMQConnector

logger = logging.getLogger(__name__)


def run_consumer(exchange: str,
                 queue: str,
                 handler: Callable[[dict], None],
                 message_broker: type[MessageBrokerConnector] = RabbitMQConnector,
                 prefetch: int = 1,
                 ) -> None:

    def _callback(channel, method, _properties, body):
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            logger.exception("failed to decode message body, dead-lettering")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return

        try:
            handler(payload)
        except Exception:
            logger.exception("handler failed, dead-lettering")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return

        channel.basic_ack(delivery_tag=method.delivery_tag)

    with message_broker(exchange=exchange, routing_key=queue) as mb:
        mb.channel.basic_qos(prefetch_count=prefetch)
        mb.channel.basic_consume(queue=queue, on_message_callback=_callback)
        logger.info(f"consuming {queue}")
        mb.channel.start_consuming()
