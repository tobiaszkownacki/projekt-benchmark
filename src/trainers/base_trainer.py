from abc import ABC, abstractmethod
import torch
from src.config import Config


class BaseTrainer(ABC):
    @abstractmethod
    def train(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        config: Config,
    ):
        pass

    @abstractmethod
    def _get_optimizer(
        self,
        optimizer_name: str,
    ) -> callable:
        pass
