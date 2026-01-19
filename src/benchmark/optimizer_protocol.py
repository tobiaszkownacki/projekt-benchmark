from typing import Protocol, runtime_checkable
import numpy as np
from src.benchmark.evaluator import ModelEvaluator

@runtime_checkable
class BenchmarkableOptimizer(Protocol):
    def step(self, evaluator: ModelEvaluator) -> bool:
        pass

class BenchmarkOptimizer:
    def __init__(self, initial_params: np.ndarray, **config):
        self.params = initial_params.copy()
        self.config = config
        # TODO: add config validation

    def step(self, evaluator: ModelEvaluator) -> bool:
        raise NotImplementedError("Implement step() in your optimizer")