from abc import ABC, abstractmethod


class Worker(ABC):
    @abstractmethod
    def start_job(self, task: dict) -> None:
        pass
