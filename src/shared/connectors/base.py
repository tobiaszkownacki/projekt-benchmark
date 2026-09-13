from abc import ABC, abstractmethod
from typing import Any


class BaseConnector(ABC):
    @abstractmethod
    def __enter__(self):
        ...
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        ...


class DatabaseConnector(BaseConnector):
    @abstractmethod
    def execute(self, query: str, params: tuple = ()) -> Any:
        ...


class MessageBrokerConnector(BaseConnector):
    @abstractmethod
    def publish(self, payload: dict) -> None:
        ...
