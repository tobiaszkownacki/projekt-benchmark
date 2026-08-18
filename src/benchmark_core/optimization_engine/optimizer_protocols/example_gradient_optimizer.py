import numpy as np
from benchmark.evaluator import ModelEvaluator
from benchmark.optimizer_protocols.benchmark_optimizer import BenchmarkOptimizer


class ExampleGradientOptimizer(BenchmarkOptimizer):
    """Example: Simple gradient descent"""

    def __init__(self, initial_params: np.ndarray, lr: float = 0.01, **config):
        super().__init__(initial_params, lr=lr, **config)
        self.lr = lr

    def step(self, evaluator: ModelEvaluator) -> bool:
        loss, grad = evaluator.evaluate_with_grad()
        self.params = self.params - self.lr * grad
        evaluator.set_params(self.params)
        return False  # Never converges on its own
