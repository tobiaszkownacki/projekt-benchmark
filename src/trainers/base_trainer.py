from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any
import torch
from src.config import BenchmarkConfig, OptimizerParams
from src.metrics.stop_metrics import MetricsTracker, StopCondition


@dataclass
class TrainingResult:
    train_losses: List[float]
    train_accuracies: List[float]
    val_losses: Optional[List[float]] = None
    val_accuracies: Optional[List[float]] = None
    final_model: Optional[torch.nn.Module] = None
    metrics_summary: Optional[dict] = None


class BaseTrainer(ABC):
    """
    Abstract class

    Further implemented optimizers can contain two possible stop metrics:
    - Gradient calculations - for gradient-based methods
    - Database reaches - for all methods (counts data samples accessed)

    Optimizers ingeriting from black-box should implement '_optimizer_step' and can
    optionally report gradient equivalents if they compute gradients internally.
    """

    def __init__(self, optimizer_name: str, optimizer_params: OptimizerParams):
        super().__init__()
        self.name = optimizer_name
        self.params = self._get_optimizer_params(optimizer_params)
        self._metrics_tracker: Optional[MetricsTracker] = None

    @property
    def metrics(self) -> Optional[MetricsTracker]:
        return self._metrics_tracker

    def train(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        config: BenchmarkConfig,
    ) -> TrainingResult:
        """
        Main training loop with unified stop metric handling.
        """
        # Metrics tracker seyup
        stop_condition = StopCondition(
            max_gradients=config.gradient_counter_stop,
            max_database_reaches=config.database_reach_limit,
            max_epochs=config.max_epochs,
        )
        self._metrics_tracker = MetricsTracker(stop_condition)

        # Delegate to implementation-specific training logic
        result = self._train_impl(model, train_dataset, config)
        result.metrics_summary = self._metrics_tracker.get_summary()

        return result

    @abstractmethod
    def _train_impl(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        config: BenchmarkConfig,
    ) -> TrainingResult:
        """Implementation-specific training logic"""
        pass

    @abstractmethod
    def _get_optimizer(self) -> Any:
        """Return the optimizer class or factory"""
        pass

    @abstractmethod
    def _get_optimizer_params(self, optimizer_params: OptimizerParams) -> dict:
        """Convert OptimizerParams to optimizer-specific dict"""
        pass

    def supports_gradients(self) -> bool:
        """Override to indicate if this trainer computes gradients"""
        return True
