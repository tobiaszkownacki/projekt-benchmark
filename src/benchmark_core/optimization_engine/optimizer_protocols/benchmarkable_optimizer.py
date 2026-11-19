from typing import Protocol, runtime_checkable

from benchmark_core.optimization_engine.evaluator import ModelEvaluator


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
                - evaluate(params) -> float: Evaluate a parameter vector
                - evaluate_with_grad(params) -> (loss, grad): Evaluate and get gradients
                - grad(params) -> ndarray: Get gradients
                - get_predictions(params) -> (predictions, targets)
                - batch_size: Number of samples in current batch
                - param_count: Total number of parameters

        Returns:
            True if optimizer has converged and wants to stop.
            False to continue optimization.

        Note:
            Metrics (database_reaches, gradient_count) are tracked
            automatically when you call evaluator methods
            The optimizer owns its parameter vectors, the evaluator does not expose them.
        """
        ...
