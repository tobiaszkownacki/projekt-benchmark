import argparse
import importlib.util
import inspect
import logging
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
from src.benchmark.optimizers import BUILTIN_OPTIMIZERS

class DummyModel(nn.Module):
    """Simple PyTorch model used for optimizer integration testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def load_custom_optimizer(path: str):
    spec = importlib.util.spec_from_file_location("custom_optimizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and hasattr(obj, "step")
            and name not in ["BenchmarkOptimizer", "NumpyBenchmarkOptimizer", "CupyBenchmarkOptimizer"]
            and not name.endswith("Dto")
        ):
            return obj
    raise ValueError(f"No valid optimizer class found in file: {path}")


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def print_status(test_name: str, passed: bool, details: str = ""):
    if passed:
        logger.info(f"PASSED {test_name}")
        if details:
            logger.info(f"   └─ {details}")
    else:
        logger.error(f"ERROR {test_name}")
        if details:
            logger.error(f"   └─ {details}")


def main():
    parser = argparse.ArgumentParser(description="Verify an optimizer against the benchmark standard.")
    parser.add_argument(
        "optimizer",
        help="Builtin optimizer name (e.g., 'adam') or path to custom .py file"
    )
    args = parser.parse_args()

    name = args.optimizer
    logger.info(f"\nStarting verification for: {name}")
    logger.info("="*50)
    all_passed = True

    # --- TEST 1: Module Loading ---
    try:
        if name in BUILTIN_OPTIMIZERS:
            opt_class = BUILTIN_OPTIMIZERS[name][0]
            print_status("Module and class loaded successfully", True, f"Found builtin class: {opt_class.__name__}")
        elif Path(name).exists():
            opt_class = load_custom_optimizer(str(name))
            print_status("Module and class loaded successfully", True, f"Found custom class: {opt_class.__name__}")
        elif Path(f"src/benchmark/optimizers/{name}").exists():
            resolved_path = Path(f"src/benchmark/optimizers/{name}")
            opt_class = load_custom_optimizer(str(resolved_path))
            print_status("Module and class loaded successfully", True, f"Found custom class in optimizers folder: {opt_class.__name__}")
        else:
            print_status("Module and class loaded successfully", False, f"Optimizer '{name}' not found as a builtin name or local file.")
            sys.exit(1)
    except Exception as e:
        print_status("Module and class loaded successfully", False, str(e))
        sys.exit(1)

    # --- TEST 2: Protocol Compliance ---
    has_step = hasattr(opt_class, "step") and callable(getattr(opt_class, "step"))
    has_get_output = hasattr(opt_class, "get_output_type") and callable(getattr(opt_class, "get_output_type"))

    if has_step and has_get_output:
        print_status("Protocol compliance check (Duck Typing)", True, "Implements required 'step' and 'get_output_type' methods")
    else:
        all_passed = False
        missing = []
        if not has_step: missing.append("step")
        if not has_get_output: missing.append("get_output_type")
        print_status("Protocol compliance check (Duck Typing)", False, f"Missing required methods: {', '.join(missing)}")

    # --- TEST 2.5: get_output_type() Method & Backend Inference ---
    xp = np
    try:
        dto_type = opt_class.get_output_type()
        print_status("Method get_output_type() is implemented", True, f"Returns DTO: {dto_type.__name__}")

        if "Cupy" in dto_type.__name__:
            try:
                import cupy as xp
                print_status("Backend inference", True, "Inferred CuPy backend from DTO")
            except ImportError:
                print_status("Backend inference", False, "DTO specifies CuPy, but cupy is not installed!")
                sys.exit(1)
        elif "Numpy" in dto_type.__name__:
            xp = np
            print_status("Backend inference", True, "Inferred NumPy backend from DTO")
        else:
            print_status("Backend inference", True, f"Unknown DTO ({dto_type.__name__}), falling back to NumPy")

    except Exception as e:
        all_passed = False
        print_status("Method get_output_type() is implemented", False, f"Missing or crashed (Did you forget @staticmethod?): {e}")

    # --- TEST 3: Constructor Signature ---
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

    # --- TEST 4: Functional Test ---
    try:
        # Dynamically choose device based on backend
        if xp.__name__ == "cupy":
            if not torch.cuda.is_available():
                print_status("Hardware Check", False, "Cannot run functional test: CuPy optimizer requires a CUDA-enabled GPU, but none was found.")
                sys.exit(1)
            device = torch.device("cuda")
            print_status("Hardware Check", True, "CUDA GPU found for CuPy optimizer")
        else:
            device = torch.device("cpu")
            print_status("Hardware Check", True, "Using CPU for NumPy optimizer")

        model = DummyModel().to(device)

        # Get PyTorch parameters and convert to the correct backend array
        raw_params = np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])
        initial_params = xp.asarray(raw_params)

        opt_instance = opt_class(initial_params=initial_params)
        print_status(f"Object initialization with initial_params ({xp.__name__} array)", True)

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

        evaluator.set_output_type(opt_class.get_output_type())
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
        opt_internal_params = getattr(opt_instance, 'params', initial_params)

        opt_internal_params_np = opt_internal_params.get() if hasattr(opt_internal_params, 'get') else opt_internal_params
        initial_params_np = raw_params

        if np.array_equal(opt_internal_params_np, initial_params_np) and grad_count > 0:
            all_passed = False
            print_status("Parameter mutation verification", False, "The parameters did not change after step() despite active gradient evaluation!")
        else:
            print_status("Parameter mutation verification", True)

        if opt_internal_params_np.dtype != initial_params_np.dtype:
            all_passed = False
            print_status("Parameter data type consistency (Dtype)", False, f"Dtype changed! Optimizer stores {opt_internal_params_np.dtype}, expected {initial_params_np.dtype}!")
        else:
            print_status("Parameter data type consistency (Dtype)", True, f"Type preserved: {opt_internal_params_np.dtype}")

    except Exception as e:
        all_passed = False
        print_status("Runtime environment execution test (Mock Run)", False, f"Execution failed: {str(e)}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()