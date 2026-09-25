from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskStatus:
    task_id: str
    task_status: str
    executor_task_id: str | None = None


class TaskRepository(ABC):
    @abstractmethod
    def mark_submitted(self, task_id: str, executor_task_id: str) -> bool: ...
    @abstractmethod
    def mark_failed(self, task_id: str, error_message: str) -> None: ...
    @abstractmethod
    def set_error(self, task_id: str, error_message: str) -> None: ...
    @abstractmethod
    def mark_completed_by_executor_id(self, executor_task_id: str) -> bool: ...
    @abstractmethod
    def get_by_executor_id(self, executor_task_id: str) -> TaskStatus | None: ...
    @abstractmethod
    def get_by_task_id(self, task_id: str) -> TaskStatus | None: ...
