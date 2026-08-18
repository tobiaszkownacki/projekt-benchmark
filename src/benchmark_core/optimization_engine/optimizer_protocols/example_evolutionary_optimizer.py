import numpy as np

from benchmark.evaluator import ModelEvaluator
from benchmark.optimizer_protocols.benchmark_optimizer import BenchmarkOptimizer


class ExampleEvolutionaryOptimizer(BenchmarkOptimizer):
    """Example: Simple (1+1) evolution strategy"""

    def __init__(self, initial_params: np.ndarray, sigma: float = 0.1, **config):
        super().__init__(initial_params, sigma=sigma, **config)
        self.sigma = sigma
        self.best_loss = float("inf")

    def step(self, evaluator: ModelEvaluator) -> bool:
        # Evaluate current
        evaluator.set_params(self.params)
        current_loss = evaluator.evaluate()

        # Try mutation
        mutant = self.params + self.sigma * np.random.randn(len(self.params))
        evaluator.set_params(mutant)
        mutant_loss = evaluator.evaluate()

        # Select better
        if mutant_loss < current_loss:
            self.params = mutant
            self.best_loss = mutant_loss
        else:
            evaluator.set_params(self.params)
            self.best_loss = current_loss

        return False  # Could add convergence check
