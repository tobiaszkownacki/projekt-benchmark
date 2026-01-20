from typing import Protocol, runtime_checkable, Dict, Any
import numpy as np
from src.benchmark.evaluator import ModelEvaluator


@runtime_checkable
class BenchmarkableOptimizer(Protocol):
    """
    Protocol for optimizers to be benchmarked
    """

    def step(self, evaluator: ModelEvaluator) -> bool:
        """
        Perform one optimization step.

        Args:
            ModelEvaluator, which provides:
                - evaluate() -> float: Get loss (forward pass)
                - evaluate_with_grad() -> (loss, grad): Get loss and gradients
                - get_params() -> np.ndarray: Get current parameters
                - set_params(params): Set new parameters
                - batch_size: Number of samples in current batch
                - param_count: Total number of parameters

        Returns:
            True if optimizer has converged and wants to stop.
            False to continue optimization.

        Note:
            Metrics (database_reaches, gradient_count) are tracked
            automatically when you call evaluator methods
        """
        ...


class BenchmarkOptimizer:
    """
    Optional base class that can be inherited by optimizers,
    provided protocol can be implemented without inheriting
    """

    def __init__(self, initial_params: np.ndarray, **config):
        """
        Initialize optimizer with model parameters

        Args:
            initial_params: Flattened model parameters as numpy array
            **config: Any additional configuration needed for the optimizer
        """
        self.params = initial_params.copy()
        self.config = config

    def step(self, evaluator: ModelEvaluator) -> bool:
        """
        TO BE OVERRIDEN
        performs one optimization

        Returns:
            True if converged, False to continue.
        """
        raise NotImplementedError("Implement step() in your optimizer")


# =============================================
# EXAMPLE IMPLEMENTATIONS (for reference)
# =============================================

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


class ExampleEvolutionaryOptimizer(BenchmarkOptimizer):
    """Example: Simple (1+1) evolution strategy"""
    
    def __init__(self, initial_params: np.ndarray, sigma: float = 0.1, **config):
        super().__init__(initial_params, sigma=sigma, **config)
        self.sigma = sigma
        self.best_loss = float('inf')
    
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