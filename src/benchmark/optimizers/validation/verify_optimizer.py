import argparse
import importlib.util
import inspect
import sys
from pathlib import Path
import traceback

# Ensure the project root is in sys.path for absolute imports
if "." not in sys.path:
    sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn

from src.benchmark.evaluator import ModelEvaluator
from src.benchmark.optimizer_protocol import BenchmarkOptimizer, BenchmarkableOptimizer
from src.benchmark.optimizers import BUILTIN_OPTIMIZERS


class DummyModel(nn.Module):
    """Simple PyTorch model used for optimizer integration testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

class NaNGradientEvaluator:
    def __init__(self, param_count: int):
        self.param_count = param_count
        self.batch_size = 4

    def evaluate(self) -> float:
        return float('nan')

    def evaluate_with_grad(self):
        bad_grad = np.zeros(self.param_count, dtype=np.float32)
        bad_grad[0] = np.nan
        bad_grad[1] = np.inf
        return float('nan'), bad_grad

    def get_params(self) -> np.ndarray:
        return np.zeros(self.param_count, dtype=np.float32)

    def set_params(self, params: np.ndarray):
        pass

def load_custom_optimizer(path: str):
    spec = importlib.util.spec_from_file_location("custom_optimizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and hasattr(obj, "step")
            and name != "BenchmarkOptimizer"
        ):
            return obj
    raise ValueError(f"No valid optimizer class found in file: {path}")


def print_status(test_name: str, passed: bool, details: str = ""):
    text = "PASSED" if passed else "ERROR"
    print(f"{text} {test_name}")
    if details:
        color = "\033[90m" if passed else "\033[91m"  # Gray for info, Red for errors
        print(f"   {color}└─ {details}\033[0m")


def main():
    parser = argparse.ArgumentParser(description="Verify an optimizer against the benchmark standard.")
    parser.add_argument(
        "optimizer",
        help="Builtin optimizer name (e.g., 'adamw') or path to custom .py file"
    )
    args = parser.parse_args()

    name = args.optimizer
    print(f"\nStarting verification for: {name}\n" + "="*50)
    all_passed = True

    try:
        if name in BUILTIN_OPTIMIZERS:
            opt_class = BUILTIN_OPTIMIZERS[name][0]
            print_status("Module and class loaded successfully", True, f"Found builtin class: {opt_class.__name__}")
        elif Path(name).exists():
            opt_class = load_custom_optimizer(str(name))
            print_status("Module and class loaded successfully", True, f"Found custom class: {opt_class.__name__}")
        else:
            print_status("Module and class loaded successfully", False, f"Optimizer '{name}' not found as a builtin name or local file.")
            print(f"Available builtins: {list(BUILTIN_OPTIMIZERS.keys())}")
            sys.exit(1)
    except Exception as e:
        print_status("Module and class loaded successfully", False, str(e))
        sys.exit(1)

    is_subclass = issubclass(opt_class, BenchmarkOptimizer)
    implements_protocol = issubclass(opt_class, BenchmarkableOptimizer)

    if is_subclass or implements_protocol:
        details = "Inherits from BenchmarkOptimizer" if is_subclass else "Implements BenchmarkableOptimizer protocol"
        print_status("Protocol compliance check", True, details)
    else:
        all_passed = False
        print_status("Protocol compliance check", False, "The class must inherit from BenchmarkOptimizer or implement BenchmarkableOptimizer")

    try:
        sig = inspect.signature(opt_class.__init__)
        params = list(sig.parameters.keys())
        if len(params) > 1 and params[1] == "initial_params":  # index 0 is 'self'
            print_status("__init__ signature accepts 'initial_params'", True)
        else:
            all_passed = False
            print_status("__init__ signature accepts 'initial_params'", False, "The first positional argument after 'self' must be 'initial_params'")
    except Exception as e:
         print_status("Checking __init__ signature", False, str(e))

    try:
        device = torch.device("cpu")
        model = DummyModel().to(device)
        initial_params = np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

        opt_instance = opt_class(initial_params=initial_params)
        print_status("Object initialization with initial_params (NumPy array)", True)

        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        db_reaches = 0
        grad_count = 0
        def dummy_metrics_callback(db_inc, grad_inc):
            nonlocal db_reaches, grad_count
            db_reaches += db_inc
            grad_count += grad_inc

        evaluator = ModelEvaluator(
            model=model,
            inputs=inputs,
            targets=targets,
            criterion=criterion,
            device=device,
            metrics_callback=dummy_metrics_callback
        )

        # Execute step
        initial_params_copy = initial_params.copy()
        result = opt_instance.step(evaluator)

        if isinstance(result, bool):
            print_status("step() method returns a boolean value (bool)", True, f"Returned: {result}")
        else:
            all_passed = False
            print_status("step() method returns a boolean value (bool)", False, f"Returned unexpected type: {type(result)}")

        if db_reaches > 0 or grad_count > 0:
            print_status("Integration with ModelEvaluator (metrics tracking)", True, f"Database reaches: {db_reaches}, Gradients calculated: {grad_count}")
        else:
            print_status("Integration with ModelEvaluator (metrics tracking)", False, "Warning: Optimizer did not call evaluate() or evaluate_with_grad()!")

        current_model_params = np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])
        if current_model_params.shape != initial_params_copy.shape:
            all_passed = False
            print_status("Parameter shape consistency (Shapes)", False, "The parameter shape in the model was altered or corrupted!")
        else:
            print_status("Parameter shape consistency (Shapes)", True)

        opt_internal_dtype = getattr(opt_instance, 'params', initial_params_copy).dtype
        if current_model_params.dtype != initial_params_copy.dtype or opt_internal_dtype != initial_params_copy.dtype:
            all_passed = False
            print_status(
                "Parameter data type consistency (Dtype)",
                False,
                f"Dtype changed! Model has {current_model_params.dtype}, but optimizer internally stores {opt_internal_dtype}!"
            )
        else:
            print_status("Parameter data type consistency (Dtype)", True, f"Type preserved: {current_model_params.dtype}")

    except Exception as e:
        all_passed = False
        print_status("Runtime environment execution test (Mock Run)", False, f"Execution failed: {str(e)}\n{traceback.format_exc()}")


    try:
        nan_evaluator = NaNGradientEvaluator(param_count=len(initial_params))
        opt_instance.step(nan_evaluator)
        print_status("Numerical stability under critical conditions (NaN/Inf Handling)", True)
    except Exception as e:
        all_passed = False
        print_status("Numerical stability under critical conditions (NaN/Inf Handling)", False, f"Crashed when encountering NaN/Inf gradients: {str(e)}")

    print("\n" + "="*50)
    if all_passed:
        print("Result: COMPLIANT! The optimizer is 100% compatible with the standard.")
        sys.exit(0)
    else:
        print("Result: ISSUES DETECTED. Fix the errors marked with  before benchmarking.")
        sys.exit(1)


if __name__ == "__main__":
    main()