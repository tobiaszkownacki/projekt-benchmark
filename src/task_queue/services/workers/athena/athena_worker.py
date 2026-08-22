import json
import logging
import os

from task_queue.services.workers.logging_config import configure_logging
from shared.interfaces.abstract_worker import Worker 

from shared.connectors.rabbitmq_connector import RabbitMQConnector
from shared.connectors.athena_connector import AthenaConnector
from shared.connectors.postgres_connector import PostGresConnector

configure_logging()
logger = logging.getLogger(__name__)

SLURM_TIME_LIMIT = "00:30:00"
MAX_EPOCHS = 10
MAX_GRADIENTS = 100000
ATHENA_REMOTE_PATH = os.environ.get("ATHENA_REMOTE_PATH")
PROJECT_DIR = f"{ATHENA_REMOTE_PATH}/projekt-benchmark"


class AthenaWorker(Worker):

    @staticmethod
    def _build_optimizer_args(optimizer_field: str) -> str:
        optimizers = [name.strip() for name in optimizer_field.split(",") if name.strip()]
        if not optimizers:
            raise ValueError("Task is missing an optimizer selection")
        if len(optimizers) == 1:
            return f"--optimizer {optimizers[0]}"
        return "--compare " + " ".join(optimizers)

    def start_job(self, task: dict) -> None:
        task_id = task["task_id"]
        try:
            optimizer_args = self._build_optimizer_args(task["optimizer"])
            job = {
                "job_name": f"job_{task_id}",
                "time": SLURM_TIME_LIMIT,
                "cpus": 1,
                "gpus": 1,
                "workdir": PROJECT_DIR,
                "run_command": (
                    f"uv run -m src.benchmark.run_benchmark "
                    f"--dataset {task['dataset']} {optimizer_args} "
                    f"--max-epochs {MAX_EPOCHS} --max-gradients {MAX_GRADIENTS} "
                    f"--task-id {task_id} --plot"
                ),
            }

            with AthenaConnector() as athena_conn:
                slurm_job_id, stderr = athena_conn.submit_job(job)

            with PostGresConnector() as db_conn:
                if stderr:
                    logger.warning(f"sbatch stderr for task_id={task_id}: {stderr}")
                    db_conn.execute(
                        "UPDATE tasks SET error_message = %s, updated_at = NOW() WHERE task_id = %s",
                        (stderr, task_id),
                    )
                db_conn.execute(
                    """
                    UPDATE tasks
                    SET executor_task_id = %s, task_status = 'running', updated_at = NOW()
                    WHERE task_id = %s
                    """,
                    (slurm_job_id, task_id),
                )
            logger.info(f"Submitted Slurm job {slurm_job_id} for task_id={task_id}")
        except Exception as exc:
            logger.exception(f"Failed to start job for task_id={task_id}")
            with PostGresConnector() as db_conn:
                db_conn.execute(
                    """
                    UPDATE tasks
                    SET task_status = 'failed', updated_at = NOW(), error_message = %s
                    WHERE task_id = %s
                    """,
                    (str(exc), task_id),
                )
            raise


def callback(rabbitmq_channel, method, properties, body):
    task_json = json.loads(body.decode("utf-8"))
    worker = AthenaWorker()

    try:
        logger.info(
            f"Received task task_id={task_json.get('task_id')} run_name={task_json.get('run_name')} "
            f"dataset={task_json.get('dataset')} optimizer={task_json.get('optimizer')}"
        )
        worker.start_job(task_json)
        rabbitmq_channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        logger.exception(
            f"Failed to execute task_id={task_json.get('task_id')} run_name={task_json.get('run_name')} "
            f"dataset={task_json.get('dataset')} optimizer={task_json.get('optimizer')} - going to the DLQ"
        )
        logger.exception(f"Exception: {e}")
        rabbitmq_channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


if __name__ == "__main__":
    with RabbitMQConnector(exchange="main-exchange", routing_key="ATHENA_WORKER_QUEUE") as rmq:
        rmq.channel.basic_qos(prefetch_count=1)
        rmq.channel.basic_consume(queue="ATHENA_WORKER_QUEUE", on_message_callback=callback)
        logger.info("Athena worker started, waiting for messages on ATHENA_WORKER_QUEUE")
        rmq.channel.start_consuming()