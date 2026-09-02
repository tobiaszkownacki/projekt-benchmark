import logging
import time

from pipeline.queue_topology import QueueTopology

logger = logging.getLogger(__name__)

class GenericPoller:

    def __init__(self,completion_rule,topology: QueueTopology,message_broker,
                 interval_s: int):
        self.completion_rule = completion_rule
        self.topology = topology
        self.message_broker = message_broker
        self.interval_s = interval_s

    def cycle(self) -> None:
        signal = self.completion_rule.on_poll()
        if signal.new_completed_tasks_with_failure:
            logger.warning(
                f"{len(signal.new_completed_tasks_with_failure)} tasks finished with error:"
                f"{signal.new_completed_tasks_with_failure}"
            )

        if not signal.new_completed_tasks_with_success:
            return
        with self.message_broker(
            exchange=self.topology.main_exchange,
            routing_key=self.topology.downloader_queue,
        ) as publisher:
            for task_id in signal.new_completed_tasks_with_success:
                publisher.publish({"task_id": task_id})
                logger.info(f"Published task_id={task_id} to downloader queue")

    def run(self) -> None:
        logger.info(f"Starting poller with interval {self.interval_s} seconds")
        while True:
            started = time.monotonic()
            try:
                self.cycle()
            except Exception as exc:
                logger.exception(f"Poller cycle failed due to {exc}")
            time.sleep(max(0.0, self.interval_s - (time.monotonic() - started)))
