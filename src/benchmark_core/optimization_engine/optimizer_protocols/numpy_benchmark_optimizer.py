from typing import Type

from src.benchmark_core.optimization_engine.evaluator import ModelEvaluator
from src.benchmark_core.optimization_engine.evaluator_dtos import EvaluatorDto, NumpyNdarrayTensorEvaluatorDto
from src.benchmark_core.optimization_engine.optimizer_protocols.benchmark_optimizer import BenchmarkOptimizer


class NumpyBenchmarkOptimizer(BenchmarkOptimizer):
    """
    Optional base class that can be inherited by optimizers,
    provided protocol can be implemented without inheriting
    """

    def step(self, evaluator: ModelEvaluator) -> bool:
        """
        TO BE OVERRIDEN
        performs one optimization

        Returns:
            True if converged, False to continue.
        """
        raise NotImplementedError("Implement step() in your optimizer")

    def get_output_type() -> Type[EvaluatorDto]:
        return NumpyNdarrayTensorEvaluatorDto
