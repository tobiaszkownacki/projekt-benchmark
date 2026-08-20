from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum, auto


class StopReason(Enum):
    GRADIENT_LIMIT = auto()
    DATABASE_REACH_LIMIT = auto()
    EPOCH_LIMIT = auto()
    CONVERGENCE = auto()
    OPTIMIZER_SIGNAL = auto()
    NONE = auto()


@dataclass
class StopCondition:
    max_gradients: Optional[int] = None  # None = unlimited
    max_database_reaches: Optional[int] = None  # None = unlimited
    max_epochs: Optional[int] = None
    convergence_threshold: Optional[float] = None

    def __post_init__(self):
        if all(
            v is None
            for v in [self.max_gradients, self.max_database_reaches, self.max_epochs]
        ):
            raise ValueError("At least one stop condition must be specified")


class MetricsTracker:
    """
    Database reaches = number of times a sample from the dataset is accessed.
    Gradient calculations = number of gradient computations.
    """

    def __init__(self, stop_condition: StopCondition):
        self.stop_condition = stop_condition
        self._gradient_count: int = 0
        self._database_reach_count: int = 0
        self._epoch_count: int = 0
        self._stop_reason: StopReason = StopReason.NONE
        self._callbacks: list[Callable[["MetricsTracker"], None]] = []

    @property
    def gradient_count(self) -> int:
        return self._gradient_count

    @property
    def database_reach_count(self) -> int:
        return self._database_reach_count

    @property
    def epoch_count(self) -> int:
        return self._epoch_count

    @property
    def stop_reason(self) -> StopReason:
        return self._stop_reason

    def record_gradients(self, count: int = 1) -> bool:
        self._gradient_count += count
        return self._check_gradient_limit()

    def record_database_reaches(self, count: int) -> bool:
        """Count database accesses (samples processed). Returns True if should stop."""
        self._database_reach_count += count
        return self._check_database_limit()

    def record_epoch(self) -> bool:
        """Count epoch completions. Returns True if should stop."""
        self._epoch_count += 1
        return self._check_epoch_limit()

    def should_stop(self) -> bool:
        """Check all stop conditions."""
        return self._stop_reason != StopReason.NONE

    def signal_optimizer_stop(self):
        """Called when optimizer signals it wants to stop early."""
        if self._stop_reason == StopReason.NONE:
            self._stop_reason = StopReason.OPTIMIZER_SIGNAL

    def _check_gradient_limit(self) -> bool:
        if (
            self.stop_condition.max_gradients is not None
            and self._gradient_count >= self.stop_condition.max_gradients
        ):
            self._stop_reason = StopReason.GRADIENT_LIMIT
            return True
        return False

    def _check_database_limit(self) -> bool:
        if (
            self.stop_condition.max_database_reaches is not None
            and self._database_reach_count >= self.stop_condition.max_database_reaches
        ):
            self._stop_reason = StopReason.DATABASE_REACH_LIMIT
            return True
        return False

    def _check_epoch_limit(self) -> bool:
        if (
            self.stop_condition.max_epochs is not None
            and self._epoch_count >= self.stop_condition.max_epochs
        ):
            self._stop_reason = StopReason.EPOCH_LIMIT
            return True
        return False

    def register_callback(self, callback: Callable[["MetricsTracker"], None]):
        """Most likely I will use it for logging purposes :)"""
        self._callbacks.append(callback)

    def _notify_callbacks(self):
        for cb in self._callbacks:
            cb(self)

    def get_summary(self) -> dict:
        return {
            "gradient_count": self._gradient_count,
            "database_reach_count": self._database_reach_count,
            "epoch_count": self._epoch_count,
            "stop_reason": self._stop_reason.name,
        }
