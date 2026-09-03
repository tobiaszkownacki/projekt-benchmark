from src.benchmark_core.optimization_engine.evaluator import ModelEvaluator
from src.benchmark_core.optimization_engine.evaluator_dtos import CupyNdarrayTensorEvaluatorDto, EvaluatorDto
from src.benchmark_core.optimization_engine.optimizer_protocols.benchmark_optimizer import BenchmarkOptimizer


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

    def get_output_type() -> type[EvaluatorDto]:
        return CupyNdarrayTensorEvaluatorDto
