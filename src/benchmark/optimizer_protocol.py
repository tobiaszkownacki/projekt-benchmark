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
