import json
import logging
import os
import time
from collections.abc import Callable

from pipeline.executor import RetryableSubmissionError
from shared.connectors.base import MessageBrokerConnector
from shared.connectors.rabbitmq import RabbitMQConnector, declare_topology
from shared.queue_topology import QueueTopology

logger = logging.getLogger(__name__)

RETRY_DELAY_SECONDS = max(1, int(os.environ.get("SUBMISSION_RETRY_DELAY_SECONDS", "30")))


def run_consumer(
    exchange: str,
    queue: str,
    handler: Callable[[dict], None],
    topology: QueueTopology,
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
        except RetryableSubmissionError as exc:
            # A scheduler admission limit is expected to clear. Delay before
            # requeueing so the worker does not hot-loop against SLURM.
            logger.warning(
                "temporary submission rejection; requeueing in %ss: %s",
                RETRY_DELAY_SECONDS,
                exc,
            )
            time.sleep(RETRY_DELAY_SECONDS)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            return
        except Exception:
            logger.exception("handler failed, dead-lettering")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return

        channel.basic_ack(delivery_tag=method.delivery_tag)

    with message_broker(exchange=exchange, routing_key=queue) as mb:
        declare_topology(mb.channel, topology)
        mb.channel.basic_qos(prefetch_count=prefetch)
        mb.channel.basic_consume(queue=queue, on_message_callback=_callback)
        logger.info(f"consuming {queue}")
        mb.channel.start_consuming()
