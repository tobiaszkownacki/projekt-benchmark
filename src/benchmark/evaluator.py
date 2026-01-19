from typing import Tuple, Callable, List
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

class ModelEvaluator:
    def __init__(
        self,
        model: Module,
        inputs: Tensor,
        targets: Tensor,
        criterion: Callable,
        device: torch.device,
        metrics_callback: Callable[[int, int], None],
    ):
        self._model = model
        self._inputs = inputs.to(device)
        self._targets = targets.to(device)
        self._criterion = criterion
        self._device = device
        self._metrics_callback = metrics_callback
        self._batch_size = targets.size(0)
        
        # Cache shapes for flattening/unflattening
        self._param_shapes = [p.shape for p in model.parameters()]
        self._param_count = sum(p.numel() for p in model.parameters())

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def param_count(self) -> int:
        return self._param_count

    def get_params(self) -> np.ndarray:
        return np.concatenate([
            p.data.cpu().numpy().flatten() 
            for p in self._model.parameters()
        ])

    def set_params(self, params: np.ndarray) -> None:
        # TODO: add validation for params shape
        idx = 0
        for p, shape in zip(self._model.parameters(), self._param_shapes):
            numel = np.prod(shape)
            # TODO: check if copy_ is the fastest way to update weights
            p.data.copy_(
                torch.from_numpy(
                    params[idx:idx + numel].reshape(shape)
                ).to(self._device)
            )
            idx += numel

    def evaluate(self) -> float:
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(self._inputs)
            loss = self._criterion(outputs, self._targets)
        
        # database reach = batch size, grads = 0
        self._metrics_callback(self._batch_size, 0)
        return loss.item()

    def evaluate_with_grad(self) -> Tuple[float, np.ndarray]:
        self._model.train()
        self._model.zero_grad()
        
        outputs = self._model(self._inputs)
        loss = self._criterion(outputs, self._targets)
        loss.backward()
        
        grad = np.concatenate([
            p.grad.cpu().numpy().flatten() 
            for p in self._model.parameters()
        ])
        
        # database reach = batch size, grads = 1
        self._metrics_callback(self._batch_size, 1)
        return loss.item(), grad

    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: Support regression tasks, currently assumes classification
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(self._inputs)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy(), self._targets.cpu().numpy()