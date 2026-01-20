from abc import ABC, abstractmethod
import torch
from src.config import BenchmarkConfig, OptimizerParams


class BaseTrainer(ABC):

    def __init__(self, optimizer_name: str, optimizer_params: OptimizerParams):
        super().__init__()
        self.name = optimizer_name
        self.params = self._get_optimizer_params(optimizer_params)

    @abstractmethod
    def train(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.TensorDataset,
        config: BenchmarkConfig,
    ):
        pass

    @abstractmethod
    def _get_optimizer(
        self,
        optimizer_name: str,
    ) -> callable:
        pass

    @abstractmethod
    def _get_optimizer_params(
        self,
        optimizer_params: OptimizerParams,
    ) -> dict:
        pass