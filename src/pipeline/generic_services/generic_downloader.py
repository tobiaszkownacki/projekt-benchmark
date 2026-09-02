import logging

from pipeline.consumer import run_consumer
from pipeline.executor import ExecutorAdapter
from pipeline.queue_topology import QueueTopology
from pipeline.task_repository import TaskRepository

logger = logging.getLogger(__name__)

class Downloader:
    def __init__(self,adapter: ExecutorAdapter,topology: QueueTopology,task_repo: TaskRepository,
                 message_broker) -> None:
        self.adapter = adapter
        self.topology = topology
        self.task_repo = task_repo
        self.message_broker = message_broker

    def handle(self, message: dict) -> None:
        task_id = message["task_id"]
        logger.info(f"Request for downloading task_id={task_id}")


        try:
            self.adapter.fetch_results(task_id, delete_after_download=False)
        except Exception as exc:
            logger.exception(f"Failed to fetch results for task_id={task_id}")
            self.task_repo.set_error(task_id, str(exc))
            raise  # task_id goes to DLQ

        logger.info(f"Downloaded results for task_id={task_id}")


    def run(self) ->None:
        run_consumer(exchange=self.topology.main_exchange,
                    queue=self.topology.downloader_queue,
                     handler=self.handle,
                     message_broker=self.message_broker
                     )
