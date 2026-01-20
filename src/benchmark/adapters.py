import numpy as np
import torch
import cma
from typing import Optional, Type

from src.benchmark.optimizer_protocol import BenchmarkOptimizer
from src.benchmark.evaluator import ModelEvaluator


class PyTorchOptimizerAdapter(BenchmarkOptimizer):
    def __init__(
        self,
        initial_params: np.ndarray,
        torch_optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        **config
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


class TorchAdamAdapter(BenchmarkOptimizer):
    """Pure NumPy implementation for benchmark"""
    # TODO: Check if this is done right
    # almost certianly it is not the same as PyTorch Adam
    def __init__(
        self,
        initial_params: np.ndarray,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        **config
    ):
        super().__init__(initial_params, **config)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Adam state
        self.m = np.zeros_like(initial_params)
        self.v = np.zeros_like(initial_params)
        self.t = 0

    def step(self, evaluator: ModelEvaluator) -> bool:
        loss, grad = evaluator.evaluate_with_grad()
        self.t += 1

        # Weight decay
        if self.weight_decay > 0:
            grad = grad + self.weight_decay * self.params

        # Adam update
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        self.params = self.params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        evaluator.set_params(self.params)

        return False


class TorchSGDAdapter(BenchmarkOptimizer):
    """Simple SGD with optional momentum."""
    def __init__(
        self,
        initial_params: np.ndarray,
        lr: float = 0.01,
        momentum: float = 0,
        weight_decay: float = 0,
        **config
    ):
        super().__init__(initial_params, **config)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = np.zeros_like(initial_params)

    def step(self, evaluator: ModelEvaluator) -> bool:
        loss, grad = evaluator.evaluate_with_grad()

        if self.weight_decay > 0:
            grad = grad + self.weight_decay * self.params

        if self.momentum > 0:
            self.velocity = self.momentum * self.velocity + grad
            self.params = self.params - self.lr * self.velocity
        else:
            self.params = self.params - self.lr * grad

        evaluator.set_params(self.params)
        return False


class CMAESAdapter(BenchmarkOptimizer):
    """CMA-ES adapter for gradient-free comparison."""
    def __init__(
        self,
        initial_params: np.ndarray,
        sigma: float = 0.5,
        population_size: Optional[int] = None,
        **config
    ):
        super().__init__(initial_params, **config)

        opts = config.get("cma_options", {}).copy()

        opts["seed"] = config.get("seed", 42)
        if population_size:
            opts["popsize"] = population_size

        # CRITICAL: Force diagonal for dimensions > 1000
        if len(initial_params) > 1000:
            opts["CMA_diagonal"] = True

        self.es = cma.CMAEvolutionStrategy(initial_params, sigma, opts)

    def step(self, evaluator: ModelEvaluator) -> bool:
        candidates = self.es.ask()

        # Evaluate each (this accumulates database_reaches)
        losses = []
        for c in candidates:
            evaluator.set_params(c)
            losses.append(evaluator.evaluate())

        # Tell results
        self.es.tell(candidates, losses)

        # Set best params
        self.params = self.es.result.xbest
        evaluator.set_params(self.params)

        return self.es.stop()


BUILTIN_OPTIMIZERS = {
    "adam": (TorchAdamAdapter, {"lr": 0.001}),
    "sgd": (TorchSGDAdapter, {"lr": 0.01}),
    "sgd_momentum": (TorchSGDAdapter, {"lr": 0.01, "momentum": 0.9}),
    "cma-es": (CMAESAdapter, {"sigma": 0.5}),
}
