from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CompletionSignal:
    new_completed_tasks_with_success: list[str]
    new_completed_tasks_with_failure: list[str]


class CompletionSource(ABC):
    @abstractmethod
    def on_poll(self) -> CompletionSignal:
        ...
