from typing import Tuple, Optional, Callable
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module


class ModelEvaluator:
    """
    Provided functions:
    - get_params() -> np.ndarray
    - set_params(params: np.ndarray) -> None
    - evaluate() -> float
    - evaluate_with_grad() -> Tuple[float, np.ndarray]
    - get_predictions() -> Tuple[np.ndarray, np.ndarray]
    - batch_size() -> int
    - param_count() -> int
    """

    def __init__(
        self,
        model: Module,
        inputs: Tensor,
        targets: Tensor,
        criterion: Callable,
        device: torch.device,
        metrics_callback: Callable[[int, int], None],  # (db_reaches, gradients)
    ):
        self._model = model
        self._inputs = inputs.to(device)
        self._targets = targets.to(device)
        self._criterion = criterion
        self._device = device
        self._metrics_callback = metrics_callback
        self._batch_size = targets.size(0)
        self._param_shapes = [p.shape for p in model.parameters()]
        self._param_count = sum(p.numel() for p in model.parameters())

    @property
    def batch_size(self) -> int:
        """Number of samples in current batch"""
        return self._batch_size

    @property
    def param_count(self) -> int:
        """Total number of model parameters"""
        return self._param_count

    def get_params(self) -> np.ndarray:
        """Get current model parameters as flat numpy array"""
        return np.concatenate(
            [p.data.cpu().numpy().flatten() for p in self._model.parameters()]
        )

    def set_params(self, params: np.ndarray) -> None:
        """Set model parameters from flat numpy array"""
        idx = 0
        for p, shape in zip(self._model.parameters(), self._param_shapes):
            numel = np.prod(shape)
            p.data.copy_(
                torch.from_numpy(params[idx : idx + numel].reshape(shape)).to(
                    self._device
                )
            )
            idx += numel

    def evaluate(self) -> float:
        """
        Evaluate current parameters on the batch (forward pass only)

        Returns:
            Loss value as float

        Effect:
            Increments database_reaches by batch_size
        """
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(self._inputs)
            loss = self._criterion(outputs, self._targets)

        # Track: forward pass = database reach
        self._metrics_callback(self._batch_size, 0)
        return loss.item()

    def evaluate_with_grad(self) -> Tuple[float, np.ndarray]:
        """
        Evaluate and compute gradients (forward + backward pass)

        Returns:
            Tuple of (loss_value, gradient_as_flat_numpy_array)

        Effect:
            Increments database_reaches by batch_size
            Increments gradient_count by 1
        """
        self._model.train()
        self._model.zero_grad()

        outputs = self._model(self._inputs)
        loss = self._criterion(outputs, self._targets)
        loss.backward()

        grad = np.concatenate(
            [p.grad.cpu().numpy().flatten() for p in self._model.parameters()]
        )

        # Track: forward+backward = database reach + gradient
        self._metrics_callback(self._batch_size, 1)
        return loss.item(), grad

    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current predictions and targets for accuracy calculation"""
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(self._inputs)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy(), self._targets.cpu().numpy()
