"""A local CPU execution backend for the benchmark.

§6 of the brief describes the multi-backend seam already present in
``shared/interfaces/`` and notes that a local CPU backend on a micro budget
would be the obvious second implementation, because it unblocks testing the web
layer without going through SLURM. This is that backend, at the smallest useful
size.

It exists for a second, more immediate reason. The repository's own
``BenchmarkRunner`` cannot run here: it imports ``src.dataset`` and ``src.config``,
neither of which is present on main -- the datasets are deliberately absent
(§5.3) and the module that indexes them went with them. Rather than invent
convergence curves for the interface to display, this runner reuses the parts
that do exist -- the real ``ModelEvaluator`` and the real NumPy optimizers -- over
public scikit-learn data.

Consequently the numbers the site displays are measured, not fabricated: the
gradient and sample counters are incremented by the project's own evaluator, on
its own terms, exactly as they would be on the cluster.
"""

import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from compat.benchmark_aliases import install as install_aliases  # noqa: E402

install_aliases()

from benchmark.evaluator import ModelEvaluator  # noqa: E402
from benchmark.evaluator_dtos import PyTorchTensorEvaluatorDto  # noqa: E402
from benchmark.optimizers.numpy.numpy_adam import NumpyAdam  # noqa: E402
from benchmark.optimizers.numpy.numpy_adamw import NumpyAdamW  # noqa: E402
from benchmark.optimizers.numpy.numpy_cmaes import NumpyCMAES  # noqa: E402
from benchmark.optimizers.numpy.numpy_des import NumpyDES  # noqa: E402
from benchmark.optimizers.numpy.numpy_differential_evolution import (  # noqa: E402
    NumpyDifferentialEvolution,
)
from benchmark.optimizers.numpy.numpy_lion import NumpyLion  # noqa: E402
from benchmark.optimizers.numpy.numpy_rmsprop import NumpyRMSProp  # noqa: E402
from benchmark.optimizers.numpy.numpy_sgd import NumpySGD  # noqa: E402

from tools.local_backend.datasets import (  # noqa: E402
    build_dataset,
    build_model,
    parameter_count,
)

RUNNER_VERSION = "local-cpu-1"

# Family is recorded rather than guessed at query time: §13.1 calls it the axis
# the whole thesis turns on, so it has to be filterable in SQL.
LOCAL_OPTIMIZERS: dict[str, tuple[Any, dict, str]] = {
    "adam": (NumpyAdam, {"lr": 0.01}, "gradient"),
    "adamw": (NumpyAdamW, {"lr": 0.01, "weight_decay": 0.01}, "gradient"),
    "lion": (NumpyLion, {"lr": 1e-3, "weight_decay": 0.01}, "gradient"),
    "rmsprop": (NumpyRMSProp, {"lr": 0.005, "alpha": 0.99}, "gradient"),
    "sgd": (NumpySGD, {"lr": 0.05}, "gradient"),
    "sgd_momentum": (NumpySGD, {"lr": 0.05, "momentum": 0.9}, "gradient"),
    "cma-es": (NumpyCMAES, {"sigma": 0.15}, "gradient_free"),
    "de": (NumpyDifferentialEvolution, {"pop_size": 20, "F": 0.8, "CR": 0.7}, "gradient_free"),
    "des": (NumpyDES, {"pop_size": 16, "sigma": 0.3}, "gradient_free"),
}


@dataclass
class StopCondition:
    max_gradient_count: Optional[int] = None
    max_database_reaches: Optional[int] = None
    max_epochs: Optional[int] = None
    max_steps: Optional[int] = None

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class LocalResult:
    optimizer_name: str
    dataset_name: str
    model_name: str
    stop_reason: str
    total_steps: int
    total_epochs: int
    wall_time_seconds: float
    final_loss: float
    final_accuracy: float
    gradient_count: int
    database_reaches: int
    seed: int
    param_count: int
    loss_history: list[float] = field(default_factory=list)
    accuracy_history: list[float] = field(default_factory=list)
    gradient_history: list[int] = field(default_factory=list)
    database_reaches_history: list[int] = field(default_factory=list)
    time_history: list[float] = field(default_factory=list)
    epoch_history: list[int] = field(default_factory=list)
    stdout: str = ""


class LocalBenchmarkRunner:
    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        stop_condition: StopCondition,
        batch_size: int = 32,
        seed: int = 2137,
    ):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.stop_condition = stop_condition
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device("cpu")

        self.dataset, self.in_features, self.classes = build_dataset(dataset_name, seed)

    def run(self, optimizer_key: str) -> LocalResult:
        optimizer_class, config, _family = LOCAL_OPTIMIZERS[optimizer_key]

        # Seeded before anything allocates, so a repeated (optimizer, seed) pair
        # reproduces. §18 flags reproducibility as a real risk on the cluster;
        # it costs nothing to get right here.
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        model = build_model(self.model_name, self.in_features, self.classes).to(self.device)
        initial_params = (
            PyTorchTensorEvaluatorDto(parameters_to_vector(model.parameters()))
            .to(optimizer_class.get_output_type())
            .data()
        )
        optimizer = optimizer_class(initial_params, **config)

        gradient_count = 0
        database_reaches = 0

        def metrics_callback(db_inc: int, grad_inc: int) -> None:
            nonlocal gradient_count, database_reaches
            database_reaches += db_inc
            gradient_count += grad_inc

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        criterion = CrossEntropyLoss()

        result = LocalResult(
            optimizer_name=optimizer_key,
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            stop_reason="EPOCH_LIMIT",
            total_steps=0,
            total_epochs=0,
            wall_time_seconds=0.0,
            final_loss=float("inf"),
            final_accuracy=0.0,
            gradient_count=0,
            database_reaches=0,
            seed=self.seed,
            param_count=parameter_count(model),
        )

        lines: list[str] = [
            f"local-cpu backend {RUNNER_VERSION}",
            f"optimizer={optimizer_key} dataset={self.dataset_name} "
            f"model={self.model_name} seed={self.seed}",
            f"parameters={result.param_count} batch_size={self.batch_size}",
            f"stop_condition={self.stop_condition.as_dict()}",
            "-" * 64,
        ]

        steps = 0
        epochs = 0
        stop_reason: Optional[str] = None
        started = time.time()

        while stop_reason is None:
            epoch_losses: list[float] = []
            correct = 0
            total = 0

            for inputs, targets in loader:
                sc = self.stop_condition
                if sc.max_gradient_count and gradient_count >= sc.max_gradient_count:
                    stop_reason = "GRADIENT_LIMIT"
                    break
                if sc.max_database_reaches and database_reaches >= sc.max_database_reaches:
                    stop_reason = "DATABASE_LIMIT"
                    break
                if sc.max_steps and steps >= sc.max_steps:
                    stop_reason = "MAX_STEPS"
                    break

                evaluator = ModelEvaluator(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    criterion=criterion,
                    device=self.device,
                    metrics_callback=metrics_callback,
                )
                evaluator.set_output_type(optimizer_class.get_output_type())

                converged = bool(optimizer.step(evaluator))
                steps += 1
                if converged:
                    stop_reason = "OPTIMIZER_CONVERGED"
                    break

                # Measured outside the evaluator so that observing the model
                # does not itself spend the participant's budget.
                with torch.no_grad():
                    outputs = model(inputs.to(self.device))
                    loss = criterion(outputs, targets.to(self.device))
                    predicted = torch.argmax(outputs, dim=1)
                    correct += int((predicted == targets.to(self.device)).sum().item())
                    total += int(targets.size(0))
                epoch_losses.append(float(loss.item()))

            if stop_reason is not None and not epoch_losses:
                break

            if epoch_losses:
                epochs += 1
                mean_loss = sum(epoch_losses) / len(epoch_losses)
                accuracy = 100.0 * correct / total if total else 0.0
                result.loss_history.append(mean_loss)
                result.accuracy_history.append(accuracy)
                result.gradient_history.append(gradient_count)
                result.database_reaches_history.append(database_reaches)
                result.time_history.append(time.time() - started)
                result.epoch_history.append(epochs)
                lines.append(
                    f"epoch {epochs:>3}  loss={mean_loss:9.4f}  acc={accuracy:6.2f}%  "
                    f"grads={gradient_count:>8}  samples={database_reaches:>9}"
                )

            if stop_reason is not None:
                break
            if self.stop_condition.max_epochs and epochs >= self.stop_condition.max_epochs:
                stop_reason = "EPOCH_LIMIT"
                break

        result.wall_time_seconds = time.time() - started
        result.total_steps = steps
        result.total_epochs = epochs
        result.gradient_count = gradient_count
        result.database_reaches = database_reaches
        result.stop_reason = stop_reason or "EPOCH_LIMIT"
        result.final_loss = result.loss_history[-1] if result.loss_history else float("inf")
        result.final_accuracy = (
            result.accuracy_history[-1] if result.accuracy_history else 0.0
        )

        lines += [
            "-" * 64,
            f"stop_reason={result.stop_reason}",
            f"final_loss={result.final_loss:.6f} final_accuracy={result.final_accuracy:.2f}%",
            f"gradients={result.gradient_count} samples={result.database_reaches}",
            f"wall_time={result.wall_time_seconds:.2f}s steps={result.total_steps}",
        ]
        result.stdout = "\n".join(lines) + "\n"
        return result
