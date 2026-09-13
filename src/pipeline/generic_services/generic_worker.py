import logging

from pipeline.consumer import run_consumer
from pipeline.executor import JobDescription
from pipeline.task_repository import TaskRepository
from shared.queue_topology import QueueTopology

logger = logging.getLogger(__name__)

class GenericWorker:

    def __init__(self,adapter,queue_topology: QueueTopology,task_repo: TaskRepository,message_broker):
        self.adapter = adapter
        self.queue_topology = queue_topology
        self.task_repo = task_repo
        self.message_broker = message_broker

    def handle(self, message: dict):
        job = JobDescription.from_message(message)
        logger.info(
            f"Received task_id={job.task_id} dataset={job.dataset} "
            f"optimizers={job.optimizers} run_name={job.run_name}"
        )
        try:
            submit_result = self.adapter.submit_job(job)
        except Exception as exc:
            logger.exception(f"Failed to submit job for task_id={job.task_id} due to {exc}")
            self.task_repo.mark_failed(job.task_id, str(exc))
            raise

        logger.info(f"task_id={job.task_id} submitted, executor_task_id={submit_result.executor_task_id}")
        self.task_repo.mark_submitted(job.task_id, submit_result.executor_task_id)

    def run(self) -> None:
        run_consumer(
            exchange=self.queue_topology.main_exchange,
            queue=self.queue_topology.worker_queue,
            handler=self.handle,
            message_broker=self.message_broker
        )


