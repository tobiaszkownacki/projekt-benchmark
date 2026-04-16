from typing import Optional

import cma
import numpy as np

from src.benchmark.evaluator import ModelEvaluator
from src.benchmark.optimizer_protocol import BenchmarkOptimizer


class CMAESAdapter(BenchmarkOptimizer):
    """CMA-ES adapter for gradient-free comparison."""

    def __init__(
        self,
        initial_params: np.ndarray,
        sigma: float = 0.5,
        population_size: Optional[int] = None,
        **config,
    ):
        super().__init__(initial_params, **config)

        opts = config.get("cma_options", {}).copy()

        opts["seed"] = config.get("seed", 42)
        if population_size:
            opts["popsize"] = population_size

        # CRITICAL: Force diagonal for dimensions > 1000
        if len(initial_params) > 1000:
            opts["CMA_diagonal"] = True

        self.es = cma.CMAEvolutionStrategy(initial_params, sigma, opts)

    def step(self, evaluator: ModelEvaluator) -> bool:
        candidates = self.es.ask()

        # Evaluate each (this accumulates database_reaches)
        losses = []
        for c in candidates:
            evaluator.set_params(c)
            losses.append(evaluator.evaluate())

        # Tell results
        self.es.tell(candidates, losses)

        # Set best params
        self.params = self.es.result.xbest
        evaluator.set_params(self.params)

        return self.es.stop()
