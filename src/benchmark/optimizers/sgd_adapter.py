import numpy as np

from src.benchmark.evaluator import ModelEvaluator
from src.benchmark.optimizer_protocol import BenchmarkOptimizer


class SGDAdapter(BenchmarkOptimizer):
    """Simple SGD with optional momentum."""

    def __init__(
        self,
        initial_params: np.ndarray,
        lr: float = 0.01,
        momentum: float = 0,
        weight_decay: float = 0,
        **config,
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
