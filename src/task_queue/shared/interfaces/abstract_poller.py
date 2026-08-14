from abc import abstractmethod, ABC


class Poller(ABC):
    def __init__(self, executor_name: str) -> None:
        self.executor_name = executor_name

    @abstractmethod
    def is_ready(self) -> bool:
        pass

    @abstractmethod
    def check_jobs_status(self) -> None:
        pass

    @abstractmethod
    def publish_finished_taks2downloader(self) -> None:
        pass
