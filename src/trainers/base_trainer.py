from abc import ABC, abstractmethod
import torch
from src.config import BenchmarkConfig, ModelType, OptimizerParams
from torch import nn


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

    def _calculate_step(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
        model_type: ModelType
    ) -> tuple[torch.Tensor, torch.Tensor]:
        match model_type:
            case ModelType.SUPERVISED:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                return loss, outputs
            case ModelType.DENSE_AE | ModelType.CONV_AE:
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                return loss, outputs
            case ModelType.VARIATIONAL_AE:
                # TODO: Implement VAE specific loss calculation
                raise NotImplementedError("VAE loss calculation not implemented yet.")
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")
