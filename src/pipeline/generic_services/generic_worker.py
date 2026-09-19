import logging

from pipeline.consumer import run_consumer
from pipeline.executor import JobDescription
from pipeline.task_repository import TaskRepository
from shared.queue_topology import QueueTopology

logger = logging.getLogger(__name__)


class GenericWorker:
    def __init__(self, adapter, queue_topology: QueueTopology, task_repo: TaskRepository, message_broker):
        self.adapter = adapter
        self.queue_topology = queue_topology
        self.task_repo = task_repo
        self.message_broker = message_broker

    def handle(self, message: dict):
        job = self._read(message)
        logger.info(
            f"Received task_id={job.task_id} dataset={job.dataset} optimizers={job.optimizers} run_name={job.run_name}"
        )
        record = self.task_repo.get_by_task_id(job.task_id)
        if record is not None and record.executor_task_id:
            logger.info(
                f"task_id={job.task_id} already runs as executor_task_id={record.executor_task_id}, "
                "so this delivery submits nothing"
            )
            return

        try:
            submit_result = self.adapter.submit_job(job)
        except Exception as exc:
            logger.exception(f"Failed to submit job for task_id={job.task_id} due to {exc}")
            self.task_repo.mark_failed(job.task_id, str(exc))
            raise

        if self.task_repo.mark_submitted(job.task_id, submit_result.executor_task_id):
            logger.info(f"task_id={job.task_id} submitted, executor_task_id={submit_result.executor_task_id}")
            return

        current = self.task_repo.get_by_task_id(job.task_id)
        if current is not None and current.executor_task_id == submit_result.executor_task_id:
            logger.info(f"task_id={job.task_id} was recorded as {submit_result.executor_task_id} by the callback")
        else:
            logger.error(
                f"task_id={job.task_id} was already {current.executor_task_id if current else 'unknown'} "
                f"when {submit_result.executor_task_id} came back, so that job runs untracked"
            )

    def _read(self, message: dict) -> JobDescription:
        """Decode, and leave a trace in the database when it cannot be done.

        The message goes to the dead-letter queue either way, and nothing
        consumes that queue. Without the row saying FAILED, a submission whose
        message the worker cannot read waits in PENDING for good.
        """
        try:
            return JobDescription.from_message(message)
        except Exception as exc:
            task_id = message.get("task_id")
            if task_id is None:
                logger.exception("Unreadable message names no task, so nothing can be marked failed")
            else:
                logger.exception(f"Cannot read the job message for task_id={task_id}")
                self.task_repo.mark_failed(str(task_id), f"unreadable job message: {exc}")
            raise

    def run(self) -> None:
        run_consumer(
            exchange=self.queue_topology.main_exchange,
            queue=self.queue_topology.worker_queue,
            handler=self.handle,
            topology=self.queue_topology,
            message_broker=self.message_broker,
        )
