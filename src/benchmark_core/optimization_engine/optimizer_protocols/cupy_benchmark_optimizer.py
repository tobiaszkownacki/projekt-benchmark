from benchmark.evaluator_dtos import EvaluatorDto
from typing import Type

from benchmark.evaluator import ModelEvaluator
from benchmark.evaluator_dtos import CupyNdarrayTensorEvaluatorDto
from benchmark.optimizer_protocols.benchmark_optimizer import BenchmarkOptimizer


class CupyBenchmarkOptimizer(BenchmarkOptimizer):
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
        return CupyNdarrayTensorEvaluatorDto
