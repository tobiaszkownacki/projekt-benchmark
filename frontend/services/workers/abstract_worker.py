from abc import abstractmethod, ABC


class Worker(ABC):
    @abstractmethod
    def start_job(self, task: dict) -> None:
        pass