from typing import List, Optional

import numpy as np
import scipy.special

from src.benchmark.evaluator import ModelEvaluator
from src.benchmark.optimizer_protocol import BenchmarkOptimizer


class DESAdapter(BenchmarkOptimizer):
    """
    Pure NumPy implementation of Differential Evolution Strategy (DES)
    with Historical Memory and Lamarckian/Darwinian boundary handling.
    """

    def __init__(
        self,
        initial_params: np.ndarray,
        Ft: float = 1.0,
        Lamarckism: bool = False,
        **config,
    ):
        super().__init__(initial_params, **config)
        self.dim = len(initial_params)
        self.Ft = Ft
        self.Lamarckism = Lamarckism

        # defaults
        self.lambda_ = config.get("lambda", 4 * self.dim)
        self.mu = config.get("mu", self.lambda_ // 2)

        # Bounding boxes
        if "bounds" in config:
            self.bounds = np.array(config["bounds"])
            self.lower = self.bounds[:, 0]
            self.upper = self.bounds[:, 1]
        else:
            self.lower = np.full(self.dim, -100.0)
            self.upper = np.full(self.dim, 100.0)

        # Recombination weights
        weights = np.log(self.mu + 1) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)

        weightsPop = np.log(self.lambda_ + 1) - np.log(np.arange(1, self.lambda_ + 1))
        self.weightsPop = weightsPop / np.sum(weightsPop)

        # Strategy Parameters
        self.cc = config.get("ccum", self.mu / (self.mu + 2))
        self.cp = config.get("cp", 1.0 / np.sqrt(self.dim))
        self.histSize = config.get(
            "history", int(np.ceil(6 + np.ceil(3 * np.sqrt(self.dim))))
        )

        # Constants
        self.chiN = np.sqrt(2) * scipy.special.gamma((self.dim + 1) / 2) / scipy.special.gamma(
            self.dim / 2
        )
        self.tol = (
            1e-6  # IMPORTANT - another diff, in R it is 1e-12, but in paper it is 1e-6
        )

        # State
        self.iter = 0
        self.histHead = -1  # incremented to 0 on first step

        # Buffers
        self.history = np.zeros((self.histSize, self.dim, self.mu))
        self.dMean = np.zeros((self.dim, self.histSize))
        self.pc = np.zeros((self.dim, self.histSize))

        self.newMean = initial_params.copy()

        # (potential) TODO original code used 0.8 of original boundries
        self.population = np.random.uniform(self.lower, self.upper, (self.lambda_, self.dim)).T
        # Insert initial guess
        self.population[:, 0] = initial_params

        self.cumMean = (self.upper + self.lower) / 2.0

        self.fitness = np.full(self.lambda_, np.inf)
        self.best_fit = np.inf
        self.worst_fit = -np.inf

        self._initialized = False

    def _bounce_back_boundary(self, x_matrix: np.ndarray) -> np.ndarray:
        """
        original: `bounceBackBoundary2`.
        Uses modulo arithmetic to bounce back into the domain.
        x_matrix shape: (dim, pop_size)
        """
        repaired = x_matrix.copy()
        range_width = self.upper - self.lower

        lower_b = self.lower[:, None]
        upper_b = self.upper[:, None]
        range_b = range_width[:, None]

        # if out of bounds
        too_low = repaired < lower_b
        too_high = repaired > upper_b

        # lower + abs(lower - x) % (upper - lower)
        repaired[too_low] = (
            lower_b[too_low]
            + np.abs(lower_b[too_low] - repaired[too_low]) % range_b[too_low]
        )
        # upper - abs(upper - x) % (upper - lower)
        repaired[too_high] = (
            upper_b[too_high]
            - np.abs(upper_b[too_high] - repaired[too_high]) % range_b[too_high]
        )

        np.nan_to_num(repaired, copy=False, nan=np.finfo(np.float64).max)
        return repaired

    def _evaluate_population(
        self, evaluator: ModelEvaluator, P: np.ndarray, P_repaired: np.ndarray
    ):
        fits = np.zeros(self.lambda_)

        if self.Lamarckism:
            # Lamarckism: Evaluate repaired individuals directly
            for i in range(self.lambda_):
                evaluator.set_params(P_repaired[:, i])
                fits[i] = evaluator.evaluate()
        else:
            # Darwinian: Evaluate valid points and penalize out of bounds
            for i in range(self.lambda_):
                # IMPORTANT, this if is DIFFERENT from orignal R but I believe to be compliant with original idea and paper
                # in R, I think, it evaluated if a point is out on ALL dimensions, here it checks if its out in even one and punishes it
                if np.array_equal(P[:, i], P_repaired[:, i]):
                    evaluator.set_params(P[:, i])
                    fits[i] = evaluator.evaluate()
                else:
                    # penalization
                    dist_sq = np.sum((P[:, i] - P_repaired[:, i]) ** 2)
                    # max seen fitness as worst.fit baseline
                    baseline = self.worst_fit if self.worst_fit > -np.inf else 1e6
                    fits[i] = baseline + dist_sq

        return fits

    def step(self, evaluator: ModelEvaluator) -> bool:
        if not self._initialized:
            # initial population eval
            pop_repaired = self._bounce_back_boundary(self.population)
            if self.Lamarckism:
                self.population = pop_repaired

            self.fitness = self._evaluate_population(evaluator, self.population, pop_repaired)
            self.worst_fit = np.max(self.fitness)

            best_idx = np.argmin(self.fitness)
            self.best_fit = self.fitness[best_idx]
            self.params = self.population[:, best_idx].copy()
            evaluator.set_params(self.params)

            self._initialized = True
            return False

        self.iter += 1
        self.histHead = (self.histHead + 1) % self.histSize

        # 1. Selection
        selection_idx = np.argsort(self.fitness)[: self.mu]
        selectedPoints = self.population[:, selection_idx]

        self.history[self.histHead] = selectedPoints * (1.0 / np.sqrt(2)) / self.Ft

        # 2. Recombination
        oldMean = self.newMean.copy()
        self.newMean = np.dot(selectedPoints, self.weights)

        popMean = np.dot(self.population, self.weightsPop)
        self.dMean[:, self.histHead] = (self.newMean - popMean) / self.Ft

        step = (self.newMean - oldMean) / self.Ft

        # 3. Update Internal Parameters
        if self.iter == 1:
            self.pc[:, self.histHead] = np.sqrt(self.mu * self.cp * (2 - self.cp)) * step
        else:
            prevHead = (self.histHead - 1) % self.histSize
            self.pc[:, self.histHead] = (1 - self.cp) * self.pc[
                :, prevHead
            ] + np.sqrt(self.mu * self.cp * (2 - self.cp)) * step

        # 4. Mutation
        limit = min(self.iter, self.histSize)

        # history momentums
        hist_samples1 = np.random.randint(0, limit, self.lambda_)  # tau_1
        hist_samples2 = np.random.randint(0, limit, self.lambda_)  # tau_2
        hist_samples3 = np.random.randint(0, limit, self.lambda_)  # tau_3 - for pc

        x1_idx = np.random.randint(0, self.mu, self.lambda_)
        x2_idx = np.random.randint(0, self.mu, self.lambda_)

        diffs = np.zeros((self.dim, self.lambda_))

        for i in range(self.lambda_):
            x1 = self.history[hist_samples1[i], :, x1_idx[i]]
            x2 = self.history[hist_samples1[i], :, x2_idx[i]]

            noise1 = np.random.randn()
            noise2 = np.random.randn()

            # IMPORTANT - difference with R
            # "Independence of the mixture components allows for the weighted summation of covariance matrices"
            part1 = np.sqrt(self.cc) * (
                (x1 - x2) + noise1 * self.dMean[:, hist_samples2[i]]
            )
            part2 = np.sqrt(1 - self.cc) * noise2 * self.pc[:, hist_samples3[i]]

            diffs[:, i] = part1 + part2

        # New population
        decay_noise = (
            self.tol
            * (1 - 2 / self.dim**2) ** (self.iter / 2)
            * np.random.randn(self.dim, self.lambda_)
            / self.chiN
        )

        self.population = self.newMean[:, None] + self.Ft * diffs + decay_noise
        np.nan_to_num(self.population, copy=False, nan=np.finfo(np.float64).max)

        # 5. Constraints and Evaluation
        pop_repaired = self._bounce_back_boundary(self.population)

        if self.Lamarckism:
            self.population = pop_repaired

        self.fitness = self._evaluate_population(evaluator, self.population, pop_repaired)

        # 6. Updates
        current_worst = np.max(self.fitness)
        if current_worst > self.worst_fit:
            self.worst_fit = current_worst

        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fit:
            self.best_fit = self.fitness[best_idx]
            self.params = (
                pop_repaired[:, best_idx]
                if not self.Lamarckism
                else self.population[:, best_idx]
            )

        self.cumMean = 0.8 * self.cumMean + 0.2 * self.newMean
        cumMeanRepaired = self._bounce_back_boundary(self.cumMean[:, None])[:, 0]
        evaluator.set_params(cumMeanRepaired)
        fn_cum = evaluator.evaluate()

        if fn_cum < self.best_fit:
            self.best_fit = fn_cum
            self.params = cumMeanRepaired.copy()

        evaluator.set_params(self.params)

        return False
