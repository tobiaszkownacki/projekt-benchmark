import logging
from pathlib import Path

from pipeline.consumer import run_consumer
from pipeline.executor import ExecutorAdapter
from pipeline.run_result import RunResult, find_manifest
from pipeline.task_repository import TaskRepository
from shared.queue_topology import QueueTopology

logger = logging.getLogger(__name__)


class Downloader:
    def __init__(
        self, adapter: ExecutorAdapter, topology: QueueTopology, task_repo: TaskRepository, message_broker
    ) -> None:
        self.adapter = adapter
        self.topology = topology
        self.task_repo = task_repo
        self.message_broker = message_broker

    def handle(self, message: dict) -> None:
        task_id = message["task_id"]
        logger.info(f"Request for downloading task_id={task_id}")

        try:
            fetched = self.adapter.fetch_results(task_id, delete_after_download=False)
        except Exception as exc:
            logger.exception(f"Failed to fetch results for task_id={task_id}")
            self.task_repo.set_error(task_id, str(exc))
            raise  # task_id goes to DLQ

        total_bytes = sum(Path(name).stat().st_size for name in fetched.files)
        self.task_repo.mark_artifacts(task_id, len(fetched.files), total_bytes)

        manifest = find_manifest(fetched.files)
        if manifest is None:
            logger.warning(f"task_id={task_id} left no result manifest; the run has artifacts but no numbers")
        else:
            self.task_repo.store_result(task_id, RunResult.from_manifest(manifest))
            logger.info(f"Stored result for task_id={task_id}")

        logger.info(f"Downloaded {len(fetched.files)} file(s) for task_id={task_id}")

    def run(self) -> None:
        run_consumer(
            exchange=self.topology.main_exchange,
            queue=self.topology.downloader_queue,
            handler=self.handle,
            message_broker=self.message_broker,
        )
