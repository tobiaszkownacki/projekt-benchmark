from typing import Type

from benchmark.evaluator import ModelEvaluator
from benchmark.evaluator_dtos import EvaluatorDto


class BenchmarkOptimizer:
    """
    Optional base class that can be inherited by optimizers,
    provided protocol can be implemented without inheriting
    """

    def __init__(self, initial_params, **config):
        """
        Initialize optimizer with model parameters

        Args:
            initial_params: Flattened model parameters as numpy array
            **config: Any additional configuration needed for the optimizer
        """
        self.config = config
        self.params = initial_params

    def step(self, evaluator: ModelEvaluator) -> bool:
        """
        TO BE OVERRIDEN
        performs one optimization

        Returns:
            True if converged, False to continue.
        """
        raise NotImplementedError("Implement step() in your optimizer")

    def get_output_type(self) -> Type[EvaluatorDto]:
        raise NotImplementedError("Implement get_output_type() in your optimizer")
