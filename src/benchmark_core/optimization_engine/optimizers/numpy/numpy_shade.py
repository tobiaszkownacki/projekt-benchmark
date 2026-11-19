import numpy as np

from benchmark_core.optimization_engine.evaluator import ModelEvaluator
from benchmark_core.optimization_engine.optimizer_protocols.numpy_benchmark_optimizer import NumpyBenchmarkOptimizer


class NumpySHADE(NumpyBenchmarkOptimizer):
    """
    Pure NumPy implementation of Success-History Based Parameter Adaptation for Differential Evolution (SHADE).
    Strategy: current-to-pbest/1/bin with external archive and history-based adaptation.
    Fully vectorized, scaling in O(N) memory allocation.
    """

    def __init__(
        self,
        initial_params: np.ndarray,
        pop_size: int = 100,
        memory_size: int = 100,
        bounds_radius: float = 2.0,
        **config,
    ):
        super().__init__(initial_params, **config)
        
        self.pop_size = pop_size
        self.memory_size = memory_size
        self.dim = len(initial_params)
        
        # p_min ensures at least 2 individuals are in the pbest pool
        self.p_min = 2.0 / self.pop_size
        # IMPORTANT - guardrail for small pop sizes where 2/N > 0.2
        self.p_max = max(self.p_min, 0.2)

        # Bounding boxes
        if "bounds" in config:
            self.bounds = np.array(config["bounds"])
            self.min_b, self.max_b = self.bounds[:, 0], self.bounds[:, 1]
        else:
            self.min_b = initial_params - bounds_radius
            self.max_b = initial_params + bounds_radius

        self.population = np.random.uniform(
            low=self.min_b,
            high=self.max_b,
            size=(self.pop_size, self.dim),
        )
        self.population[0] = initial_params

        # Buffers
        self.M_CR = np.full(self.memory_size, 0.5)
        self.M_F = np.full(self.memory_size, 0.5)
        self.archive = np.empty((0, self.dim), dtype=float)
        
        # State
        self.memory_index = 0
        self.fitness = np.full(self.pop_size, np.inf)
        self._initialized = False

    def step(self, evaluator: ModelEvaluator) -> bool:
        if not self._initialized:
            for i in range(self.pop_size):
                self.fitness[i] = evaluator.evaluate(self.population[i])
            
            best_idx = np.argmin(self.fitness)
            self.params = self.population[best_idx]
            self._initialized = True
            return False

        N = self.pop_size
        D = self.dim
        
        # 1. Hoisted Archive Union
        # Avoids O(N^2) memory reallocation by doing this once per generation
        if len(self.archive) > 0:
            union = np.vstack([self.population, self.archive])
        else:
            union = self.population
        M = len(union)

        # 2. Vectorized Parameter Generation
        r_idx = np.random.randint(0, self.memory_size, size=N)
        
        # CR = randn(M_CR, 0.1) clipped to [0, 1]
        CR = np.random.normal(self.M_CR[r_idx], 0.1)
        CR = np.clip(CR, 0.0, 1.0)
        
        # F = randc(M_F, 0.1), F <= 0 resampled, F > 1 truncated
        F = self.M_F[r_idx] + 0.1 * np.random.standard_cauchy(size=N)
        invalid_mask = F <= 0.0
        while np.any(invalid_mask):
            num_invalid = np.sum(invalid_mask)
            F[invalid_mask] = self.M_F[r_idx][invalid_mask] + 0.1 * np.random.standard_cauchy(size=num_invalid)
            invalid_mask = F <= 0.0
        F = np.minimum(F, 1.0)
        
        # p_i = rand[p_min, 0.2]
        p = np.random.uniform(self.p_min, self.p_max, size=N)
        p_nums = np.maximum(2, (p * N).astype(int))

        # 3. Vectorized Index Selection
        i_idx = np.arange(N)
        
        # r1 from Population (r1 != i)
        offset1 = np.random.randint(1, N, size=N)
        r1 = (i_idx + offset1) % N
        
        # r2 from Union (r2 != i, r2 != r1) 
        # IMPORTANT - rank exclusion avoids while-loop retries and guarantees strict distinctness
        lo = np.minimum(i_idx, r1)
        hi = np.maximum(i_idx, r1)
        raw = np.random.randint(0, M - 2, size=N)
        r2 = raw + (raw >= lo) + (raw >= (hi - 1))

        # pbest target selection
        sorted_idx = np.argsort(self.fitness)
        pbest_idx = np.array([np.random.choice(sorted_idx[:p_nums[i]]) for i in range(N)])

        # 4. Vectorized current-to-pbest/1 Mutation
        x_i = self.population
        x_pbest = self.population[pbest_idx]
        x_r1 = self.population[r1]
        x_r2 = union[r2]
        
        F_vec = F[:, None]
        # v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
        v = x_i + F_vec * (x_pbest - x_i) + F_vec * (x_r1 - x_r2)
        
        # Boundary Correction (bounce back halfway to parent)
        below = v < self.min_b
        above = v > self.max_b
        v = np.where(below, (self.min_b + x_i) / 2.0, v)
        v = np.where(above, (self.max_b + x_i) / 2.0, v)

        # 5. Vectorized Binomial Crossover
        CR_vec = CR[:, None]
        cross_mask = np.random.rand(N, D) < CR_vec
        j_rand = np.random.randint(0, D, size=N)
        cross_mask[np.arange(N), j_rand] = True
        
        # u = v if cross_mask else x_i
        u_trials = np.where(cross_mask, v, x_i)

        # 6. Evaluation and Selection
        S_CR, S_F, delta_f = [], [], []
        new_archive_items = []
        
        for i in range(N):
            f_trial = evaluator.evaluate(u_trials[i])
            
            # x_i,G+1 = u_i,G if f(u_i,G) <= f(x_i,G)
            if f_trial <= self.fitness[i]:
                # strictly improving trials feed the archive and historical memory
                if f_trial < self.fitness[i]:
                    new_archive_items.append(self.population[i].copy())
                    S_CR.append(CR[i])
                    S_F.append(F[i])
                    delta_f.append(self.fitness[i] - f_trial)
                
                self.population[i] = u_trials[i]
                self.fitness[i] = f_trial

        # 7. Archive and Memory Management
        if new_archive_items:
            self.archive = np.vstack((self.archive, new_archive_items))
            
        # random deletion to keep |A| <= |P|
        if len(self.archive) > self.pop_size:
            keep_idx = np.random.choice(len(self.archive), size=self.pop_size, replace=False)
            self.archive = self.archive[keep_idx]

        if len(S_CR) > 0 and len(S_F) > 0:
            S_CR_arr = np.array(S_CR)
            S_F_arr = np.array(S_F)
            
            # w_k = delta_f_k / sum(delta_f)
            weights = np.array(delta_f) / np.sum(delta_f)
            
            # M_CR: weighted arithmetic mean
            self.M_CR[self.memory_index] = np.sum(weights * S_CR_arr)
            # M_F: weighted Lehmer mean
            self.M_F[self.memory_index] = np.sum(weights * (S_F_arr ** 2)) / np.sum(weights * S_F_arr)
            
            self.memory_index = (self.memory_index + 1) % self.memory_size

        # Update global best for the protocol
        best_idx = np.argmin(self.fitness)
        self.params = self.population[best_idx]
        
        return False