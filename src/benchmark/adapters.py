import numpy as np
import torch
import cma
from typing import Optional, Type

from src.benchmark.optimizer_protocol import BenchmarkOptimizer
from src.benchmark.evaluator import ModelEvaluator


class PyTorchOptimizerAdapter(BenchmarkOptimizer):
    def __init__(
        self,
        initial_params: np.ndarray,
        torch_optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        **config
    ):
        super().__init__(initial_params, **config)
        self.torch_optimizer_class = torch_optimizer_class
        self.torch_config = config
        self._optimizer = None
        self._param_tensor = None

    def step(self, evaluator: ModelEvaluator) -> bool:
        if self._optimizer is None:
            self._param_tensor = torch.nn.Parameter(
                torch.from_numpy(self.params).float()
            )
            # TODO: I have not idea how to implement that yet...
            # many attempts failed
            pass

        loss, grad = evaluator.evaluate_with_grad()

        lr = self.torch_config.get("lr", 0.001)
        self.params = evaluator.get_params() - lr * grad
        evaluator.set_params(self.params)

        return False


class AdamAdapter(BenchmarkOptimizer):
    """Pure NumPy implementation for benchmark"""

    # TODO: Check if this is done right
    # almost certianly it is not the same as PyTorch Adam
    def __init__(
        self,
        initial_params: np.ndarray,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        **config
    ):
        super().__init__(initial_params, **config)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Adam state
        self.m = np.zeros_like(initial_params)
        self.v = np.zeros_like(initial_params)
        self.t = 0

    def step(self, evaluator: ModelEvaluator) -> bool:
        loss, grad = evaluator.evaluate_with_grad()
        self.t += 1

        # Weight decay
        if self.weight_decay > 0:
            grad = grad + self.weight_decay * self.params

        # Adam update
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        self.params = self.params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        evaluator.set_params(self.params)

        return False


class AdamWAdapter(BenchmarkOptimizer):
    """Pure NumPy implementation of AdamW (Decoupled Weight Decay)."""

    def __init__(
        self,
        initial_params: np.ndarray,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        **config
    ):
        super().__init__(initial_params, **config)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Adam state
        self.m = np.zeros_like(initial_params)
        self.v = np.zeros_like(initial_params)
        self.t = 0

    def step(self, evaluator: ModelEvaluator) -> bool:
        loss, grad = evaluator.evaluate_with_grad()
        self.t += 1

        # Adam update 
        # Grads are NOT modified by weight decay here
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        step_val = m_hat / (np.sqrt(v_hat) + self.eps)

        # Decoupled weight decay: params = params - lr * (step + weight_decay * params)
        if self.weight_decay > 0:
            self.params = self.params - self.lr * (step_val + self.weight_decay * self.params)
        else:
            self.params = self.params - self.lr * step_val

        evaluator.set_params(self.params)

        return False
    

class LionAdapter(BenchmarkOptimizer):
    """Pure NumPy implementation of Lion Optimizer (Chen et al., 2023)."""

    def __init__(
        self,
        initial_params: np.ndarray,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
        **config
    ):
        super().__init__(initial_params, **config)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay

        # State: Lion only needs to track momentum (exp_avg)
        self.m = np.zeros_like(initial_params)

    def step(self, evaluator: ModelEvaluator) -> bool:
        loss, grad = evaluator.evaluate_with_grad()

        # 1. Parameter update (interp momentum & gradient)
        # c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        c_t = self.beta1 * self.m + (1 - self.beta1) * grad

        # update = sign(c_t) + weight_decay * params
        update = np.sign(c_t)
        if self.weight_decay > 0:
            update = update + self.weight_decay * self.params

        # params = params - lr * update
        self.params = self.params - self.lr * update

        # 2. Momentum update
        # m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
        self.m = self.beta2 * self.m + (1 - self.beta2) * grad

        evaluator.set_params(self.params)
        return False


class RMSPropAdapter(BenchmarkOptimizer):
    """Pure NumPy implementation of RMSProp optimizer."""

    def __init__(
        self,
        initial_params: np.ndarray,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        **config
    ):
        super().__init__(initial_params, **config)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum

        # State
        self.v = np.zeros_like(initial_params)  # Square avg
        self.b = np.zeros_like(initial_params)  # Momentum buffer

    def step(self, evaluator: ModelEvaluator) -> bool:
        loss, grad = evaluator.evaluate_with_grad()

        if self.weight_decay > 0:
            grad = grad + self.weight_decay * self.params

        # v_t = alpha * v_{t-1} + (1 - alpha) * g^2
        self.v = self.alpha * self.v + (1 - self.alpha) * (grad**2)

        # Standard RMSProp update: g / (sqrt(v) + eps)
        avg = np.sqrt(self.v) + self.eps
        
        if self.momentum > 0:
            # b_t = momentum * b_{t-1} + g / avg
            self.b = self.momentum * self.b + grad / avg
            step_val = self.b
        else:
            step_val = grad / avg

        self.params = self.params - self.lr * step_val
        evaluator.set_params(self.params)

        return False


class SGDAdapter(BenchmarkOptimizer):
    """Simple SGD with optional momentum."""

    def __init__(
        self,
        initial_params: np.ndarray,
        lr: float = 0.01,
        momentum: float = 0,
        weight_decay: float = 0,
        **config
    ):
        super().__init__(initial_params, **config)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = np.zeros_like(initial_params)

    def step(self, evaluator: ModelEvaluator) -> bool:
        loss, grad = evaluator.evaluate_with_grad()

        if self.weight_decay > 0:
            grad = grad + self.weight_decay * self.params

        if self.momentum > 0:
            self.velocity = self.momentum * self.velocity + grad
            self.params = self.params - self.lr * self.velocity
        else:
            self.params = self.params - self.lr * grad

        evaluator.set_params(self.params)
        return False


class CMAESAdapter(BenchmarkOptimizer):
    """CMA-ES adapter for gradient-free comparison."""

    def __init__(
        self,
        initial_params: np.ndarray,
        sigma: float = 0.5,
        population_size: Optional[int] = None,
        **config
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
        pop_size: int = 15,         # Standard seems to be 10 * dim, but 15-50 is good for testing
        F: float = 0.8,             # Differential weight (0.5 - 1.0)
        CR: float = 0.7,            # Crossover probability (0.0 - 1.0)
        bounds_radius: float = 2.0, # Used if explicit bounds aren't in config
        **config
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
            size=(self.pop_size, self.dim)
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
        
        v_donor = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
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


BUILTIN_OPTIMIZERS = {
    "adam": (AdamAdapter, {"lr": 0.001}),
    "adamw": (AdamWAdapter, {"lr": 0.001, "weight_decay": 0.01}),
    "lion": (LionAdapter, {"lr": 1e-4, "weight_decay": 0.01}),
    "rmsprop": (RMSPropAdapter, {"lr": 0.01, "alpha": 0.99}),
    "sgd": (SGDAdapter, {"lr": 0.01}),
    "sgd_momentum": (SGDAdapter, {"lr": 0.01, "momentum": 0.9}),
    "cma-es": (CMAESAdapter, {"sigma": 0.5}),
    "de": (DifferentialEvolutionAdapter, {"pop_size": 15, "F": 0.8, "CR": 0.7}),
    }


# TODO: Possible variation on DE (https://github.com/Valdecy/Metaheuristic-Differential_Evolution/blob/master/Python-MH-Differential%20Evolution.py)
