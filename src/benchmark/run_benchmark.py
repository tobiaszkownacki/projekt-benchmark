"""
Simple runner script with very basic argparse

Usage:
    python -m src.benchmark.run_benchmark --dataset digits --optimizer my_optimizer
    or: uv run -m src.benchmark.run_benchmark --dataset digits --optimizer my_optimizer
    
or with comparison and plotting:
    python -m src.benchmark.run_benchmark --dataset wine_quality --optimizer adam sgd cma-es --max-epochs 10 --max-gradients 100000 --plot
    or: uv run -m src.benchmark.run_benchmark --dataset wine_quality --optimizer adam sgd cma-es --max-epochs 10 --max-gradients 100000 --plot
"""

import argparse
import importlib
import sys
from pathlib import Path

from src.benchmark import BenchmarkRunner, StopCondition
from src.benchmark.optimizers import BUILTIN_OPTIMIZERS
from src.config import ALLOWED_DATASETS
from src.plotting.benchmark_analyzer import BenchmarkAnalyzer


def load_custom_optimizer(path: str):
    spec = importlib.util.spec_from_file_location("custom_optimizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the optimizer class (first class that has 'step' method)
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and hasattr(obj, "step")
            and name != "BenchmarkOptimizer"
        ):
            return obj

    raise ValueError(f"No optimizer class found in {path}")


def main():
    parser = argparse.ArgumentParser(description="Run optimizer benchmarks")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=ALLOWED_DATASETS,
    )
    parser.add_argument(
        "--optimizer", nargs="+",help="Path to custom optimizer file or builtin name"
    )
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

    args = parser.parse_args()

    # Build stop condition
    stop_condition = StopCondition(
        max_gradient_count=args.max_gradients,
        max_database_reaches=args.max_db_reaches,
        max_epochs=args.max_epochs,
    )

    runner = BenchmarkRunner(
        dataset_name=args.dataset,
        stop_condition=stop_condition,
        batch_size=args.batch_size,
        random_seed=args.seed,
    )

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

    results = runner.compare(optimizers)

    if args.plot:
        analyzer = BenchmarkAnalyzer(output_dir=args.plot_dir)
        analyzer.plot_results(results)



if __name__ == "__main__":
    main()
