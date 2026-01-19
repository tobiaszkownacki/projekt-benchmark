import numpy as np
from src.benchmark.optimizer_protocol import BenchmarkOptimizer
from src.benchmark.evaluator import ModelEvaluator

class SimpleSGD(BenchmarkOptimizer):
    def __init__(self, initial_params: np.ndarray, lr: float = 0.01):
        # Initialize base class (stores self.params)
        super().__init__(initial_params)
        self.lr = lr

    def step(self, evaluator: ModelEvaluator) -> bool:
        loss, grads = evaluator.evaluate_with_grad()
        current_params = evaluator.get_params()
        new_params = current_params - (self.lr * grads)
        evaluator.set_params(new_params)

        return False