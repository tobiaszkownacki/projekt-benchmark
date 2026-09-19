from abc import ABC, abstractmethod
from dataclasses import dataclass

from shared.run_result import RunResult


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
    @abstractmethod
    def mark_running_by_executor_id(self, executor_task_id: str) -> bool: ...
    @abstractmethod
    def store_result(self, task_id: str, result: RunResult) -> None: ...
    @abstractmethod
    def mark_artifacts(self, task_id: str, files: int, total_bytes: int) -> None: ...
