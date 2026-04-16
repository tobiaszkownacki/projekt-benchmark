"""
BenchmarkRunner - Runs any optimizer against datasets and collects metrics

Abstract:
1. Load dataset and model from predefined configurations
2. Wrap them in ModelEvaluator
3. Run the optimizer until stop condition
4. Collect and report results
TODO: 5. Plotting
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Type, Callable
from enum import Enum, auto
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from src.benchmark.evaluator import ModelEvaluator
from src.benchmark.optimizer_protocol import BenchmarkableOptimizer
from src.logging import Log


class StopReason(Enum):
    GRADIENT_LIMIT = auto()
    DATABASE_LIMIT = auto()
    EPOCH_LIMIT = auto()
    OPTIMIZER_CONVERGED = auto()
    MAX_STEPS = auto()


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    optimizer_name: str
    dataset_name: str

    # Stop info
    stop_reason: StopReason
    total_steps: int
    total_epochs: int
    wall_time_seconds: float

    # Metrics
    final_loss: float
    final_accuracy: float
    gradient_count: int
    database_reaches: int

    # History
    loss_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimizer": self.optimizer_name,
            "dataset": self.dataset_name,
            "stop_reason": self.stop_reason.name,
            "steps": self.total_steps,
            "epochs": self.total_epochs,
            "wall_time": self.wall_time_seconds,
            "final_loss": self.final_loss,
            "final_accuracy": self.final_accuracy,
            "gradient_count": self.gradient_count,
            "database_reaches": self.database_reaches,
        }


@dataclass
class StopCondition:
    """When to stop the benchmark."""

    max_gradient_count: Optional[int] = None
    max_database_reaches: Optional[int] = None
    max_epochs: Optional[int] = None
    max_steps: Optional[int] = None  # steps = optimizer.step() calls

    def __post_init__(self):
        if all(
            v is None
            for v in [
                self.max_gradient_count,
                self.max_database_reaches,
                self.max_epochs,
                self.max_steps,
            ]
        ):
            raise ValueError("At least one stop condition required")


class BenchmarkRunner:
    """
    Runs optimizers against datasets and collects metrics

    Usage:
        runner = BenchmarkRunner(
            dataset_name="digits",
            stop_condition=StopCondition(max_gradient_count=5000)
        )
        result = runner.run(MyOptimizer, lr=0.01)
    """

    def __init__(
        self,
        dataset_name: str,
        stop_condition: StopCondition,
        batch_size: int = 32,
        random_seed: int = 2137,
        log_interval: int = 10,
        device: Optional[str] = None,
    ):
        from src.dataset import (
            DATA_SETS,
        )  # Import is here due to circular dependency error

        if dataset_name not in DATA_SETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.dataset_name = dataset_name
        self.stop_condition = stop_condition
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.log_interval = log_interval

        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = DATA_SETS[dataset_name]["data_set"]()
        self.model_factory = DATA_SETS[dataset_name]["model"]

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def run(
        self,
        optimizer_class: Type[BenchmarkableOptimizer],
        optimizer_name: Optional[str] = None,
        **optimizer_config,
    ) -> BenchmarkResult:
        """
        Run benchmark for a single optimizer

        Args:
            optimizer_class: The optimizer class to benchmark
            optimizer_name: Name for logging (defaults to class name)
            **optimizer_config: Arguments passed to optimizer.__init__

        Returns:
            BenchmarkResult with all metrics
        """
        name = optimizer_name or optimizer_class.__name__

        model = self.model_factory()
        model.to(self.device)

        initial_params = np.concatenate(
            [p.data.cpu().numpy().flatten() for p in model.parameters()]
        )

        optimizer = optimizer_class(initial_params, **optimizer_config)

        # metric counters
        gradient_count = 0
        database_reaches = 0

        def metrics_callback(db_inc: int, grad_inc: int):
            nonlocal gradient_count, database_reaches
            database_reaches += db_inc
            gradient_count += grad_inc

        train_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        criterion = CrossEntropyLoss()

        log = Log(output_file=f"benchmark-{name}-{self.dataset_name}.csv")

        loss_history = []
        accuracy_history = []
        step_count = 0
        epoch_count = 0
        stop_reason = None

        start_time = time.time()

        # main loop
        while stop_reason is None:
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0

            for inputs, targets in train_loader:
                # Checking stop conditions BEFORE step
                if (
                    self.stop_condition.max_gradient_count
                    and gradient_count >= self.stop_condition.max_gradient_count
                ):
                    stop_reason = StopReason.GRADIENT_LIMIT
                    break
                if (
                    self.stop_condition.max_database_reaches
                    and database_reaches >= self.stop_condition.max_database_reaches
                ):
                    stop_reason = StopReason.DATABASE_LIMIT
                    break
                if (
                    self.stop_condition.max_steps
                    and step_count >= self.stop_condition.max_steps
                ):
                    stop_reason = StopReason.MAX_STEPS
                    break

                # Create evaluator for this batch
                evaluator = ModelEvaluator(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    criterion=criterion,
                    device=self.device,
                    metrics_callback=metrics_callback,
                )

                converged = optimizer.step(evaluator)
                step_count += 1

                if converged:
                    stop_reason = StopReason.OPTIMIZER_CONVERGED
                    break

                with torch.no_grad():
                    outputs = model(inputs.to(self.device))
                    loss = criterion(outputs, targets.to(self.device))
                    _, predicted = torch.max(outputs, 1)
                    correct = (predicted == targets.to(self.device)).sum().item()

                epoch_losses.append(loss.item())
                epoch_correct += correct
                epoch_total += targets.size(0)

                # Log periodically
                log.add_number_of_samples(targets.size(0))
                log.log(round(loss.item(), 4), self.log_interval)

            if stop_reason is not None:
                break

            epoch_count += 1

            if epoch_losses:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                accuracy = 100 * epoch_correct / epoch_total
                loss_history.append(avg_loss)
                accuracy_history.append(accuracy)

                print(
                    f"Epoch {epoch_count}: loss={avg_loss:.4f}, "
                    f"acc={accuracy:.2f}%, grads={gradient_count}, "
                    f"db_reaches={database_reaches}"
                )

            if (
                self.stop_condition.max_epochs
                and epoch_count >= self.stop_condition.max_epochs
            ):
                stop_reason = StopReason.EPOCH_LIMIT
                break

        wall_time = time.time() - start_time
        log.save_to_csv()

        final_loss = loss_history[-1] if loss_history else float("inf")
        final_accuracy = accuracy_history[-1] if accuracy_history else 0.0

        return BenchmarkResult(
            optimizer_name=name,
            dataset_name=self.dataset_name,
            stop_reason=stop_reason or StopReason.EPOCH_LIMIT,
            total_steps=step_count,
            total_epochs=epoch_count,
            wall_time_seconds=wall_time,
            final_loss=final_loss,
            final_accuracy=final_accuracy,
            gradient_count=gradient_count,
            database_reaches=database_reaches,
            loss_history=loss_history,
            accuracy_history=accuracy_history,
        )

    def compare(
        self,
        optimizers: Dict[str, tuple],  # name -> (class, config_dict)
    ) -> Dict[str, BenchmarkResult]:
        """
        Run benchmark for multiple optimizers and compare.

        Args:
            optimizers: Dict mapping name to (optimizer_class, config_dict)

        Returns:
            Dict mapping name to BenchmarkResult
        """
        results = {}
        for name, (cls, config) in optimizers.items():
            print(f"\n{'='*50}")
            print(f"Running: {name}")
            print("=" * 50)
            results[name] = self.run(cls, optimizer_name=name, **config)

        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(
            f"{'Optimizer':<20} {'Loss':>10} {'Acc':>8} {'Grads':>10} {'DB Reach':>12}"
        )
        print("-" * 60)
        for name, r in results.items():
            print(
                f"{name:<20} {r.final_loss:>10.4f} {r.final_accuracy:>7.2f}% "
                f"{r.gradient_count:>10} {r.database_reaches:>12}"
            )

        return results
