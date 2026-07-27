from abc import abstractmethod, ABC


class Poller(ABC):
    def __init__(self, executor_id: str) -> None:
        self.executor_id = executor_id

    @abstractmethod
    def check_status(self) -> bool:
        pass

    @abstractmethod
    def check_running_jobs(self) -> None:
        pass

    @abstractmethod
    def check_finalized_jobs(self) -> None:
        pass
