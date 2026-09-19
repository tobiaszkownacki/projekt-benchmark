"""
Simple runner script with very basic argparse

Usage:
    python -m benchmark_core.optimization_engine.run_benchmark --dataset digits --optimizer my_optimizer
    or: uv run -m benchmark_core.optimization_engine.run_benchmark --dataset digits --optimizer my_optimizer

or with comparison and plotting:
    python -m benchmark_core.optimization_engine.run_benchmark --dataset wine_quality --optimizer adam sgd cma-es \
        --max-epochs 10 --max-gradients 100000 --plot
    or: uv run -m benchmark_core.optimization_engine.run_benchmark --dataset wine_quality --optimizer adam sgd cma-es \
        --max-epochs 10 --max-gradients 100000 --plot
"""

import argparse
import importlib
import sys
from pathlib import Path

from benchmark_core.datasets import DATA_SETS
from benchmark_core.optimization_engine import BenchmarkRunner, StopCondition
from benchmark_core.optimization_engine.optimizers import BUILTIN_OPTIMIZERS
from benchmark_core.plotting.benchmark_analyzer import BenchmarkAnalyzer
from shared.run_result import RunResult


def load_custom_optimizer(path: str):
    spec = importlib.util.spec_from_file_location("custom_optimizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the optimizer class (first class that has 'step' method)
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and hasattr(obj, "step") and name != "BenchmarkOptimizer":
            return obj

    raise ValueError(f"No optimizer class found in {path}")


def main():
    parser = argparse.ArgumentParser(description="Run optimizer benchmarks")
    parser.add_argument(
        "--dataset",
        required=True,
        # The registry is the only list of datasets there is; a hard-coded one
        # would drift from what the runner can actually load.
        choices=sorted(DATA_SETS),
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["default"],
        help="Model architecture(s) to use (default: ['default'])",
    )
    parser.add_argument("--optimizer", nargs="+", help="Path to custom optimizer file or builtin name")
    parser.add_argument("--max-gradients", type=int, default=5000)
    parser.add_argument("--max-db-reaches", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate benchmark plots after run",
    )
    parser.add_argument(
        "--plot-dir",
        default="reports/model_analysis",
        help="Directory where plots are written",
    )
    parser.add_argument("--task-id", default=None, help="Task id - outputs land in reports/task_<id>")
    args = parser.parse_args()
    report_dir = f"reports/task_{args.task_id}" if args.task_id else "reports"
    if not args.optimizer:
        parser.print_help()
        return

    optimizers = {}
    for name in args.optimizer:
        if name in BUILTIN_OPTIMIZERS:
            optimizers[name] = BUILTIN_OPTIMIZERS[name]
        elif Path(name).exists():
            cls = load_custom_optimizer(name)
            optimizers[name] = (cls, {})
        else:
            print(f"Optimizer not found: {name}")
            print(f"Available: {list(BUILTIN_OPTIMIZERS.keys())}")
            sys.exit(1)

    # Build stop condition
    stop_condition = StopCondition(
        max_gradient_count=args.max_gradients,
        max_database_reaches=args.max_db_reaches,
        max_epochs=args.max_epochs,
    )

    all_results = {}

    for current_model in args.model:
        runner = BenchmarkRunner(
            dataset_name=args.dataset,
            model_name=current_model,
            stop_condition=stop_condition,
            batch_size=args.batch_size,
            random_seed=args.seed,
            report_dir=report_dir,
        )

        model_results = runner.compare(optimizers)

        for opt_name, result in model_results.items():
            run_identifier = f"{current_model}_{opt_name}"
            result.optimizer_name = run_identifier
            all_results[run_identifier] = result

    if args.plot:
        analyzer = BenchmarkAnalyzer(output_dir=report_dir)
        analyzer.plot_results(all_results)

    # The numbers have to leave the node as a file: nothing on the cluster can
    # reach the database, and the download phase is what writes the row.
    if args.task_id and len(all_results) == 1:
        result = next(iter(all_results.values()))
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        RunResult.from_benchmark_result(result).write_manifest(Path(report_dir))


if __name__ == "__main__":
    main()
