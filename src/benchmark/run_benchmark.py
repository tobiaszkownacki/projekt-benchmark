"""
Simple runner script with very basic argparse

Usage:
    python -m src.benchmark.run_benchmark --dataset digits --optimizer my_optimizer
or with comparison:
    python -m src.benchmark.run_benchmark --dataset digits --compare adam sgd cma-es
"""

import argparse
import importlib
import sys
from pathlib import Path

from src.benchmark import BenchmarkRunner, StopCondition
from src.benchmark.adapters import BUILTIN_OPTIMIZERS


def load_custom_optimizer(path: str):
    spec = importlib.util.spec_from_file_location("custom_optimizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find the optimizer class (first class that has 'step' method)
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and hasattr(obj, 'step') and name != 'BenchmarkOptimizer':
            return obj
    
    raise ValueError(f"No optimizer class found in {path}")


def main():
    parser = argparse.ArgumentParser(description="Run optimizer benchmarks")
    parser.add_argument("--dataset", required=True, 
                        choices=["digits", "cifar10", "heart_disease", "wine_quality"])
    parser.add_argument("--optimizer", help="Path to custom optimizer file or builtin name")
    parser.add_argument("--compare", nargs="+", help="Compare multiple builtins")
    parser.add_argument("--max-gradients", type=int, default=5000)
    parser.add_argument("--max-db-reaches", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    
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
    
    if args.compare:
        # Compare multiple builtins
        optimizers = {}
        for name in args.compare:
            if name in BUILTIN_OPTIMIZERS:
                optimizers[name] = BUILTIN_OPTIMIZERS[name]
            else:
                print(f"Unknown builtin optimizer: {name}")
                print(f"Available: {list(BUILTIN_OPTIMIZERS.keys())}")
                sys.exit(1)
        
        runner.compare(optimizers)
    
    elif args.optimizer:
        # Run single optimizer
        if args.optimizer in BUILTIN_OPTIMIZERS:
            cls, config = BUILTIN_OPTIMIZERS[args.optimizer]
            result = runner.run(cls, **config)
        elif Path(args.optimizer).exists():
            cls = load_custom_optimizer(args.optimizer)
            result = runner.run(cls)
        else:
            print(f"Optimizer not found: {args.optimizer}")
            sys.exit(1)
        
        print("\nResult:")
        for k, v in result.to_dict().items():
            print(f"  {k}: {v}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()