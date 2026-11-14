import numpy as np

from benchmark_core.optimization_engine.evaluator import ModelEvaluator
from benchmark_core.optimization_engine.optimizer_protocols.numpy_benchmark_optimizer import NumpyBenchmarkOptimizer


class NumpyCSO(NumpyBenchmarkOptimizer):
    """
    Pure NumPy implementation of the Competitive Swarm Optimizer (CSO).
    Designed for large-scale optimization without historical memory.
    """

    def __init__(
        self,
        initial_params: np.ndarray,
        pop_size: int = 200,
        phi: float | None = None,
        bounds_radius: float = 2.0,
        **config,
    ):
        super().__init__(initial_params, **config)
        
        # Swarm size must be even to allow for exact pairwise competition
        self.pop_size = pop_size if pop_size % 2 == 0 else pop_size + 1
        self.dim = len(initial_params)
        
        # Dynamically set the social factor (phi) if not explicitly provided
        if phi is None:
            if self.pop_size <= 100:
                self.phi = 0.0
            else:
                # Based on the empirical lower bound recommendation: 0.14 * log(m) - 0.30
                empirical_phi = 0.14 * np.log10(self.pop_size) - 0.30
                self.phi = max(0.0, empirical_phi)
        else:
            self.phi = phi

        # 1. Define Bounds
        if "bounds" in config:
            self.bounds = np.array(config["bounds"])
            self.min_b, self.max_b = self.bounds[:, 0], self.bounds[:, 1]
        else:
            self.min_b = initial_params - bounds_radius
            self.max_b = initial_params + bounds_radius

        # 2. Initialize Population and Velocity
        self.population = np.random.uniform(
            low=self.min_b,
            high=self.max_b,
            size=(self.pop_size, self.dim),
        )
        self.population[0] = initial_params
        
        self.velocity = np.zeros((self.pop_size, self.dim), dtype=float)

        self.fitness = np.full(self.pop_size, np.inf)
        self._initialized = False

    def _ensure_bounds(self, trials: np.ndarray) -> np.ndarray:
        """Standard boundary clamping."""
        return np.clip(trials, self.min_b, self.max_b)

    def step(self, evaluator: ModelEvaluator) -> bool:
        # Initial Evaluation (First Step only)
        if not self._initialized:
            for i in range(self.pop_size):
                self.fitness[i] = evaluator.evaluate(self.population[i])
            
            best_idx = np.argmin(self.fitness)
            self.params = self.population[best_idx].copy()
            self._initialized = True
            return False

        half = self.pop_size // 2
        
        # 1. Random Pairwise Allocation
        indices = np.random.permutation(self.pop_size)
        group1 = indices[:half]
        group2 = indices[half:]
        
        # 2. Competition
        # mask is True where group1 particle is better (lower fitness) than group2 particle
        mask = self.fitness[group1] <= self.fitness[group2]
        
        winners = np.where(mask, group1, group2)
        losers = np.where(mask, group2, group1)
        
        # 3. Learning Strategy Preparation
        mean_X = np.mean(self.population, axis=0)
        
        R1 = np.random.rand(half, self.dim)
        R2 = np.random.rand(half, self.dim)
        R3 = np.random.rand(half, self.dim)
        
        # 4. Vectorized Velocity and Position Updates for Losers
        # V_l(t+1) = R1*V_l(t) + R2*(X_w(t) - X_l(t)) + phi*R3*(mean_X - X_l(t))
        self.velocity[losers] = (
            R1 * self.velocity[losers] + 
            R2 * (self.population[winners] - self.population[losers]) + 
            self.phi * R3 * (mean_X - self.population[losers])
        )
        
        self.population[losers] = self.population[losers] + self.velocity[losers]
        self.population[losers] = self._ensure_bounds(self.population[losers])
        
        # 5. Evaluation (Evaluate ONLY the losers)
        for idx in losers:
            self.fitness[idx] = evaluator.evaluate(self.population[idx])

        # Update global best for the protocol tracking
        best_idx = np.argmin(self.fitness)
        self.params = self.population[best_idx].copy()
        
        return False