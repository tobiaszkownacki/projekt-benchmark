import numpy as np

from src.benchmark.evaluator import ModelEvaluator
from src.benchmark.optimizer_protocol import BenchmarkOptimizer


class AdamAdapter(BenchmarkOptimizer):
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
