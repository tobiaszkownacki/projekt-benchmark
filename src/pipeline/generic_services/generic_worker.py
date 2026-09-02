import logging

from pipeline.consumer import run_consumer
from pipeline.executor import JobDescription
from pipeline.queue_topology import QueueTopology
from pipeline.task_repository import TaskRepository

logger = logging.getLogger(__name__)

class GenericWorker:

    def __init__(self,adapter,queue_topology: QueueTopology,task_repo: TaskRepository,message_broker):
        self.adapter = adapter
        self.queue_topology = queue_topology
        self.task_repo = task_repo
        self.message_broker = message_broker

    def handle(self,job_description: JobDescription):
        logger.info(
            f"Received task_id={job_description.task_id} dataset={job_description.dataset} "
            f"optimizers={job_description.optimizers} run_name={job_description.run_name}"
        )
        try:
            submit_result = self.adapter.submit_job(job_description)
        except Exception as exc:
            logger.exception(f"Failed to submit job for task_id={job_description.task_id} due to {exc}")
            self.task_repo.mark_failed(job_description.task_id, str(exc))
            raise

        logger.info(f"task_id={job_description.task_id} submitted, executor_task_id={submit_result.executor_task_id}")
        self.task_repo.mark_submitted(job_description.task_id, submit_result.executor_task_id)

    def run(self) -> None:
        run_consumer(
            exchange=self.queue_topology.main_exchange,
            queue=self.queue_topology.worker_queue,
            handler=self.handle,
            message_broker=self.message_broker
        )


