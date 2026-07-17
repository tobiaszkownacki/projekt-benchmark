from typing import Protocol, runtime_checkable

from benchmark.evaluator import ModelEvaluator


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
