import cma
import numpy as np

from benchmark_core.optimization_engine.evaluator import ModelEvaluator
from benchmark_core.optimization_engine.optimizer_protocols import NumpyBenchmarkOptimizer


class NumpyCMAES(NumpyBenchmarkOptimizer):
    """CMA-ES adapter for gradient-free comparison."""

    def __init__(
        self,
        initial_params: np.ndarray,
        sigma: float = 0.5,
        population_size: int | None = None,
        **config,
    ):
        super().__init__(initial_params, **config)

        opts = config.get("cma_options", {}).copy()

        # CMA-ES samples from its own RNG. Pinning it to a constant made every
        # run over the same model identical however the caller seeded NumPy, so
        # a sweep over eight seeds measured one sample eight times. Deriving the
        # default from the global NumPy RNG keeps a seeded caller reproducible
        # and gives an unseeded one a different draw each time.
        seed = config.get("seed")
        if seed is None:
            seed = int(np.random.randint(1, 2**31 - 1))
        opts["seed"] = seed
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

        return bool(self.es.stop())
