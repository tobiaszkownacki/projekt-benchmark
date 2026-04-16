import numpy as np

from src.benchmark.evaluator import ModelEvaluator
from src.benchmark.optimizer_protocol import BenchmarkOptimizer


class DifferentialEvolutionAdapter(BenchmarkOptimizer):
    """
    Pure NumPy implementation of Differential Evolution (Storn & Price, 1997).
    Strategy: DE/rand/1/bin
    TODO: change to DE/local-to-best/1/bin as in https://hal.ii.pw.edu.pl/details

    With trial, no pure random restart

    Sources:
    https://machinelearningmastery.com/differential-evolution-from-scratch-in-python/
    https://www.researchgate.net/publication/227242104_Differential_Evolution_-_A_Simple_and_Efficient_Heuristic_for_Global_Optimization_over_Continuous_Spaces
    """

    def __init__(
        self,
        initial_params: np.ndarray,
        pop_size: int = 15,  # Standard seems to be 10 * dim, but 15-50 is good for testing
        F: float = 0.8,  # Differential weight (0.5 - 1.0)
        CR: float = 0.7,  # Crossover probability (0.0 - 1.0)
        bounds_radius: float = 2.0,  # Used if explicit bounds aren't in config
        **config,
    ):
        super().__init__(initial_params, **config)
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.dim = len(initial_params)

        # 1. Define Bounds
        # If bounds is in config, expect list of (min, max). Else use radius around init
        if "bounds" in config:
            self.bounds = np.array(config["bounds"])
            self.min_b, self.max_b = self.bounds[:, 0], self.bounds[:, 1]
        else:
            # Create heuristic bounds around the initial guess
            self.min_b = initial_params - bounds_radius
            self.max_b = initial_params + bounds_radius

        # 2. Initialize Population
        # We include the initial_params as one member to ensure we don't regress
        self.population = np.random.uniform(
            low=self.min_b,
            high=self.max_b,
            size=(self.pop_size, self.dim),
        )
        self.population[0] = initial_params

        # Track fitness of population (initially infinite)
        self.fitness = np.full(self.pop_size, np.inf)
        self._initialized = False

    def _ensure_bounds(self, trials):
        """Handle boundary constraints (Clip method)."""
        return np.clip(trials, self.min_b, self.max_b)

    def step(self, evaluator: ModelEvaluator) -> bool:
        # Initial Evaluation (First Step only)
        if not self._initialized:
            for i in range(self.pop_size):
                evaluator.set_params(self.population[i])
                self.fitness[i] = evaluator.evaluate()

            # Set the "current best" for the benchmark protocol to track
            best_idx = np.argmin(self.fitness)
            self.params = self.population[best_idx]
            evaluator.set_params(self.params)
            self._initialized = True
            return False

        # Generation Step:

        # 1. Mutation (DE/rand/1)
        # Vectorized selection of r1, r2, r3
        indices = np.arange(self.pop_size)

        # TODO: Make it compliant with r1 != r2 != r3 != target from orignal paper
        r1 = np.random.randint(0, self.pop_size, self.pop_size)
        r2 = np.random.randint(0, self.pop_size, self.pop_size)
        r3 = np.random.randint(0, self.pop_size, self.pop_size)

        # Verification mask to ensure distinctness (simple retry logic for collisions is common)
        # For simplicity/speed in this snippet, we proceed. In strict DE, we resample collisions.

        v_donor = self.population[r1] + self.F * (
            self.population[r2] - self.population[r3]
        )
        v_donor = self._ensure_bounds(v_donor)

        # 2. Binomial Crossover
        cross_mask = np.random.rand(self.pop_size, self.dim) < self.CR

        # Guaranteed parameter: Ensure at least one dimension comes from donor (as per paper)
        # TODO: find out why
        j_rand = np.random.randint(0, self.dim, self.pop_size)
        cross_mask[np.arange(self.pop_size), j_rand] = True

        u_trials = np.where(cross_mask, v_donor, self.population)

        # 3. Greedy selection with trial evaluation
        for i in range(self.pop_size):
            evaluator.set_params(u_trials[i])
            f_trial = evaluator.evaluate()

            if f_trial <= self.fitness[i]:
                self.population[i] = u_trials[i]
                self.fitness[i] = f_trial

        # 4. Update global best for the protocol
        best_idx = np.argmin(self.fitness)
        self.params = self.population[best_idx]
        evaluator.set_params(self.params)

        return False
