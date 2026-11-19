"""
This wraps the model, data batch, and loss function
providing a simple interface allowing optimizers to:
1. Evaluate supplied parameters (forward pass)
2. Get gradients (backward pass)
3. Keep model parameters hidden from the optimizer

Metrics are tracked AUTOMATICALLY, the optimizer doesn't need to do anything.
"""

from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.utils import vector_to_parameters

from .evaluator_dtos import PyTorchTensorEvaluatorDto
from .evaluator_dtos.evaluator_dto import T


class ModelEvaluator:
    """
    This class wraps a model, data batch, and loss function, providing a unified interface
    for optimizers to interact with the model's parameters and evaluate its performance.
    It automatically tracks metrics such as database reaches and gradient computations.

    Key functionalities provided:
    - `evaluate(params: object) -> float`: Evaluates supplied parameters and returns the loss.
    - `evaluate_with_grad(params: object) -> Tuple[float, object]`: Performs a forward and backward pass, returning loss
      and gradients (DTO-converted).
    - `get_predictions(params: object) -> Tuple[object, object]`: Retrieves model predictions and targets (DTO-converted).
    - `batch_size() -> int`: Returns the number of samples in the current batch.
    - `param_count() -> int`: Returns the total number of model parameters.
    """

    def __init__(
        self,
        model: Module,
        criterion: Callable,
        device: torch.device,
        metrics_callback: Callable[[int, int], None],
        inputs: Tensor | None = None,
        targets: Tensor | None = None,
    ):
        """
        Initializes the ModelEvaluator.

        Args:
            model: The neural network model (torch.nn.Module).
            criterion: The loss function (callable).
            device: The device to run the computations on (e.g., 'cpu' or 'cuda').
            metrics_callback: A callback function to track metrics (e.g., database reaches, gradient count).
            inputs: Optional initial input data batch (torch.Tensor).
            targets: Optional initial target data batch (torch.Tensor).
        """
        self._model = model
        self._criterion = criterion
        self._device = device
        self._metrics_callback = metrics_callback
        self._inputs = None
        self._targets = None
        self._batch_size = 0
        if inputs is not None and targets is not None:
            self.set_batch(inputs, targets)
        self._param_shapes = [p.shape for p in model.parameters()]
        self._param_count = sum(p.numel() for p in model.parameters())

    def set_batch(self, inputs: Tensor, targets: Tensor) -> None:
        self._inputs = inputs.to(self._device)
        self._targets = targets.to(self._device)
        self._batch_size = targets.size(0)

    def set_output_type(self, output_type: type[T]):
        self.type = output_type

    @property
    def batch_size(self) -> int:
        """Number of samples in current batch"""
        return self._batch_size

    @property
    def param_count(self) -> int:
        """Total number of model parameters"""
        return self._param_count

    def _load_params(self, params: object) -> None:
        """
        Maps a supplied flattened vector onto the model parameter tensors.

        Args:
            params: Flattened parameters in the optimizer's declared DTO representation.
        """
        params_torch_flat = self.type(params).to(PyTorchTensorEvaluatorDto, device=self._device)
        vector_to_parameters(params_torch_flat.data().to(self._device), self._model.parameters())

    def evaluate(self, params: object) -> float:
        """
        Evaluate supplied parameters on the batch (forward pass only).

        Returns:
            Loss value as float

        Effect:
            Increments database_reaches by batch_size
        """
        self._load_params(params)
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(self._inputs)
            loss = self._criterion(outputs, self._targets)

        # Track: forward pass = database reach
        self._metrics_callback(self._batch_size, 0)
        return loss.item()

    def evaluate_with_grad(self, params: object) -> tuple[float, object]:
        """
        Evaluates supplied parameters on the batch (forward pass) and computes
        gradients (backward pass).

        Returns:
            Tuple[float, object]: A tuple containing the loss value and the
                                   DTO-converted flattened gradient vector.
        Effect:
            Increments database_reaches by batch_size.
            Increments gradient_count by 1.
        """
        self._load_params(params)
        self._model.train()
        self._model.zero_grad()

        outputs = self._model(self._inputs)
        loss = self._criterion(outputs, self._targets)
        loss.backward()

        grad = PyTorchTensorEvaluatorDto(self._get_gradients_as_vector(self._model))

        # Track: forward+backward = database reach + gradient
        self._metrics_callback(self._batch_size, 1)
        return loss.item(), grad.to(self.type).data()

    def grad(self, params: object) -> object:
        """
        Computes gradients for supplied parameters on the batch (performs forward and backward pass).

        Returns:
            object: The DTO-converted flattened gradient vector.

        Effect:
            Increments database_reaches by batch_size.
            Increments gradient_count by 1.
        """
        self._load_params(params)
        self._model.train()
        self._model.zero_grad()

        outputs = self._model(self._inputs)
        loss = self._criterion(outputs, self._targets)
        loss.backward()

        grad = PyTorchTensorEvaluatorDto(self._get_gradients_as_vector(self._model))

        # Track: forward+backward = database reach + gradient
        self._metrics_callback(self._batch_size, 1)
        return grad.to(self.type).data()

    def get_predictions(self, params: object) -> tuple[object, object]:
        """
        Retrieves the model's predictions and the actual targets for the current batch,
        both converted to the specified output DTO type.

        Returns:
            Tuple[object, object]: A tuple containing the DTO-converted predictions
                                and DTO-converted targets.
        """
        self._load_params(params)
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(self._inputs)
            _, predicted = torch.max(outputs, 1)

        # Track: forward pass = database reach
        self._metrics_callback(self._batch_size, 0)

        return (
            PyTorchTensorEvaluatorDto(predicted).to(self.type).data(),
            PyTorchTensorEvaluatorDto(self._targets).to(self.type).data(),
        )

    def _get_gradients_as_vector(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Extracts and concatenates the gradients of all trainable parameters in the model
        into a single flattened tensor.

        Args:
            model: The PyTorch model from which to extract gradients.

        Returns:
            torch.Tensor: A 1D tensor containing all gradients.
                        Returns zeros for parameters that do not have gradients.
        """
        grads = []
        for param in model.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
                else:
                    grads.append(torch.zeros_like(param).view(-1))

        return torch.cat(grads)
