from abc import abstractmethod, ABC


class Worker(ABC):
    @abstractmethod
    def execute(self):
        pass