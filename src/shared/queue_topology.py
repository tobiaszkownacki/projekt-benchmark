from dataclasses import dataclass


@dataclass(frozen=True)
class QueueTopology:
    executor_name: str
    main_exchange: str = "main_exchange"
    dlx_exchange: str = "dlx_exchange"
    @property
    def worker_queue(self):
        return f"{self.executor_name}_worker_queue"
    @property
    def downloader_queue(self):
        return f"{self.executor_name}_downloader_queue"
    @property
    def worker_dlq_queue(self):
        return f"dlq_{self.worker_queue}"
    @property
    def downloader_dlq_queue(self):
        return f"dlq_{self.downloader_queue}"
    @property
    def worker_dlq_routing_key(self):
        return f"failed_{self.executor_name}_worker_task"
    @property
    def downloader_dlq_routing_key(self):
        return f"failed_{self.executor_name}_downloader_task"
