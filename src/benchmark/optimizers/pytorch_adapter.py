from typing import Type

import numpy as np
import torch

from src.benchmark.evaluator import ModelEvaluator
from src.benchmark.optimizer_protocol import BenchmarkOptimizer


class PyTorchOptimizerAdapter(BenchmarkOptimizer):
    def __init__(
        self,
        initial_params: np.ndarray,
        torch_optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        **config,
    ):
        super().__init__(initial_params, **config)
        self.torch_optimizer_class = torch_optimizer_class
        self.torch_config = config
        self._optimizer = None
        self._param_tensor = None

    def step(self, evaluator: ModelEvaluator) -> bool:
        if self._optimizer is None:
            self._param_tensor = torch.nn.Parameter(
                torch.from_numpy(self.params).float()
            )
            # TODO: I have not idea how to implement that yet...
            # many attempts failed
            pass

        loss, grad = evaluator.evaluate_with_grad()

        lr = self.torch_config.get("lr", 0.001)
        self.params = evaluator.get_params() - lr * grad
        evaluator.set_params(self.params)

        return False