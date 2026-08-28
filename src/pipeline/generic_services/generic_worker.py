import logging
from abc import ABC

from pipeline import repository
from pipeline.connectors import Connectors
from pipeline.consumer import run_consumer
from pipeline.executor import JobDescription
from pipeline.queue_topology import QueueTopology

logger = logging.getLogger(__name__)

class GenericWorker:

    def __init__(self,adapter,queue_topology: QueueTopology,connectors: Connectors):
        self.adapter = adapter
        self.queue_topology = queue_topology
        self.connectors = connectors

    def handle(self,job_description: JobDescription):

        logger.info(f"Received task_id={job_description.task_id} for dataset={job_description.dataset} with optimizers={job_description.optimizers} and run_name={job_description.run_name}")

        try:
            submit_result = self.adapter.submit_job(job_description)
        except Exception as exc:
            logger.exception(f"Failed to submit job for task_id={job_description.task_id} due to {exc}")
            self.connectors.db.execute("UPDATE tasks SET status = 'FAILED' WHERE task_id = %s", (job_description.task_id,))
            with self.connectors.db() as db:
                repository.set_error(db, job_description.task_id, str(exc))
            raise

        with self.connectors.db() as db:
            repository.set_submitted(db, job_description.task_id, submit_result.executor_task_id, submit_result.std_out)

    def run(self) -> None:
        run_consumer(
            exchange=self.queue_topology.main_exchange,
            queue=self.queue_topology.worker_queue,
            handler=self.handle,
            message_broker=self.connectors.message_broker
        )


