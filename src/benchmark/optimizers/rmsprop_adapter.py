import numpy as np

from src.benchmark.evaluator import ModelEvaluator
from src.benchmark.optimizer_protocol import BenchmarkOptimizer


class RMSPropAdapter(BenchmarkOptimizer):
    """Pure NumPy implementation of RMSProp optimizer."""

    def __init__(
        self,
        initial_params: np.ndarray,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        **config
    ):
        super().__init__(initial_params, **config)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum

        # State
        self.v = np.zeros_like(initial_params)  # Square avg
        self.b = np.zeros_like(initial_params)  # Momentum buffer

    def step(self, evaluator: ModelEvaluator) -> bool:
        loss, grad = evaluator.evaluate_with_grad()

        if self.weight_decay > 0:
            grad = grad + self.weight_decay * self.params

        # v_t = alpha * v_{t-1} + (1 - alpha) * g^2
        self.v = self.alpha * self.v + (1 - self.alpha) * (grad**2)

        # Standard RMSProp update: g / (sqrt(v) + eps)
        avg = np.sqrt(self.v) + self.eps
        
        if self.momentum > 0:
            # b_t = momentum * b_{t-1} + g / avg
            self.b = self.momentum * self.b + grad / avg
            step_val = self.b
        else:
            step_val = grad / avg

        self.params = self.params - self.lr * step_val
        evaluator.set_params(self.params)

        return False
