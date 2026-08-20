import json
import logging
import os

from shared.interfaces.abstract_downloader import Downloader
from task_queue.services.downloaders.logging_config import configure_logging

from shared.connectors.rabbitmq_connector import RabbitMQConnector
from shared.connectors.athena_connector import AthenaConnector
from shared.connectors.postgres_connector import PostGresConnector

configure_logging()
logger = logging.getLogger(__name__)

ATHENA_REMOTE_PATH = os.environ.get("ATHENA_REMOTE_PATH")
PROJECT_DIR = f"{ATHENA_REMOTE_PATH}/projekt-benchmark"
LOCAL_DOWNLOAD_DIR = os.environ.get("LOCAL_DOWNLOAD_DIR", "/downloads")


class AthenaDownloader(Downloader):

    def download(self, task_id: str, delete_after_download: bool = False) -> None:
        remote_dir = f"{PROJECT_DIR}/reports/task_{task_id}"
        local_dir = f"{LOCAL_DOWNLOAD_DIR}/{task_id}"

        try:
            with AthenaConnector() as athena_conn:
                downloaded = athena_conn.download_results(remote_dir, local_dir)

                if not downloaded:
                    message = f"No files found under {remote_dir}"
                    logger.warning(f"No files found under {remote_dir}")
                    with PostGresConnector() as db_conn:
                        db_conn.execute(
                            "UPDATE tasks SET error_message = %s, updated_at = NOW() WHERE task_id = %s",
                            (message, task_id),
                        )
                    return

                logger.info(f"Downloaded {len(downloaded)} file(s) for task_id={task_id} to {local_dir}")

                if delete_after_download:
                    athena_conn.ssh(f"rm -rf {remote_dir}")
                    logger.info(f"Deleted {remote_dir} on Athena for task_id={task_id}")
        except Exception as exc:
            logger.exception(f"Failed to download results for task_id={task_id}")
            with PostGresConnector() as db_conn:
                db_conn.execute(
                    "UPDATE tasks SET error_message = %s, updated_at = NOW() WHERE task_id = %s",
                    (str(exc), task_id),
                )
            raise


def callback(rabbitmq_channel, method, properties, body):
    payload = json.loads(body.decode("utf-8"))
    task_id = payload.get("task_id")
    downloader = AthenaDownloader()

    try:
        logger.info(f"Received download request for task_id={task_id}")
        downloader.download(task_id)
        rabbitmq_channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception:
        logger.exception(f"Failed to download task_id={task_id} - going to the DLQ")
        rabbitmq_channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


if __name__ == "__main__":
    with RabbitMQConnector(exchange="main-exchange", routing_key="ATHENA_DOWNLOADER_QUEUE") as rmq:
        rmq.channel.basic_qos(prefetch_count=1)
        rmq.channel.basic_consume(queue="ATHENA_DOWNLOADER_QUEUE", on_message_callback=callback)
        logger.info("Athena downloader started, waiting for messages on ATHENA_DOWNLOADER_QUEUE")
        rmq.channel.start_consuming()
