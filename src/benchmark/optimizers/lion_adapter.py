import numpy as np

from src.benchmark.evaluator import ModelEvaluator
from src.benchmark.optimizer_protocol import BenchmarkOptimizer


class LionAdapter(BenchmarkOptimizer):
    """Pure NumPy implementation of Lion Optimizer (Chen et al., 2023)."""

    def __init__(
        self,
        initial_params: np.ndarray,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
        **config,
    ):
        super().__init__(initial_params, **config)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay

        # State: Lion only needs to track momentum (exp_avg)
        self.m = np.zeros_like(initial_params)

    def step(self, evaluator: ModelEvaluator) -> bool:
        loss, grad = evaluator.evaluate_with_grad()

        # 1. Parameter update (interp momentum & gradient)
        # c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        c_t = self.beta1 * self.m + (1 - self.beta1) * grad

        # update = sign(c_t) + weight_decay * params
        update = np.sign(c_t)
        if self.weight_decay > 0:
            update = update + self.weight_decay * self.params

        # params = params - lr * update
        self.params = self.params - self.lr * update

        # 2. Momentum update
        # m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
        self.m = self.beta2 * self.m + (1 - self.beta2) * grad

        evaluator.set_params(self.params)
        return False
