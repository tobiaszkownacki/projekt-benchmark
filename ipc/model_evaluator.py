from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.utils import vector_to_parameters


class ModelEvaluator:
    def __init__(
        self,
        model: Module,
        criterion: Callable,
        device: torch.device,
        metrics_callback: Callable[[int, int], None],
        inputs: Tensor | None = None,
        targets: Tensor | None = None,
    ):
        self._model = model
        self._criterion = criterion
        self._device = device
        self._metrics_callback = metrics_callback
        self._inputs: Tensor | None = None
        self._targets: Tensor | None = None
        self._batch_size = 0
        self._param_shapes = [p.shape for p in model.parameters()]
        self._param_count = sum(p.numel() for p in model.parameters())

        if inputs is not None and targets is not None:
            self.set_batch(inputs, targets)

    def set_batch(self, inputs: Tensor, targets: Tensor) -> None:
        self._inputs = inputs.to(self._device)
        self._targets = targets.to(self._device)
        self._batch_size = targets.size(0)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def param_count(self) -> int:
        return self._param_count

    def _load_params(self, params: Tensor) -> None:
        vector_to_parameters(params.to(self._device), self._model.parameters())

    def evaluate(self, params: Tensor) -> float:
        self._load_params(params)
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(self._inputs)
            loss = self._criterion(outputs, self._targets)

        self._metrics_callback(self._batch_size, 0)
        return loss.item()

    def evaluate_with_grad(self, params: Tensor) -> tuple[float, Tensor]:
        self._load_params(params)
        self._model.train()
        self._model.zero_grad()

        outputs = self._model(self._inputs)
        loss = self._criterion(outputs, self._targets)
        loss.backward()

        grad = self._get_gradients_as_vector(self._model)

        self._metrics_callback(self._batch_size, 1)
        return loss.item(), grad.detach()

    def grad(self, params: Tensor) -> Tensor:
        self._load_params(params)
        self._model.train()
        self._model.zero_grad()

        outputs = self._model(self._inputs)
        loss = self._criterion(outputs, self._targets)
        loss.backward()

        grad = self._get_gradients_as_vector(self._model)

        self._metrics_callback(self._batch_size, 1)
        return grad.detach()

    def get_predictions(self, params: Tensor) -> tuple[Tensor, Tensor]:
        self._load_params(params)
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(self._inputs)
            _, predicted = torch.max(outputs, 1)

        self._metrics_callback(self._batch_size, 0)
        return predicted, self._targets

    def _get_gradients_as_vector(self, model: torch.nn.Module) -> torch.Tensor:
        grads = []
        for param in model.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
                else:
                    grads.append(torch.zeros_like(param).view(-1))

        return torch.cat(grads)
