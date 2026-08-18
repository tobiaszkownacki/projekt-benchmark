# Optimizers Implementations

## 1. Overview

This directory contains concrete implementations (adapters) of various optimization algorithms, designed to be benchmarked by the system. These adapters translate generic optimizer concepts into the specific `EvaluatorDto` types required by the benchmark framework, allowing them to interact seamlessly with the `ModelEvaluator`.

Each file typically contains an adapter class for a specific optimization algorithm (e.g., Adam, SGD, CMA-ES), inheriting from a suitable base class defined in `src/benchmark/optimizer_protocols/`.

## 2. Structure

-   **Adapter Files** (e.g., `adam_adapter.py`, `sgd_adapter.py`, `cmaes_adapter.py`): Each of these files defines an optimizer class (e.g., `AdamAdapter`, `SGDAdapter`) that implements the optimization logic. They typically inherit from `BenchmarkOptimizer`, `NumpyBenchmarkOptimizer`, or `CupyBenchmarkOptimizer` from the `optimizer_protocols` package.
-   **`registry.py`**: This is a crucial file that acts as a central lookup for all available optimizers in the system. It maps a string identifier (e.g., "adam", "sgd", "cmaes") to the corresponding optimizer class. This registry is used by the benchmark runner to instantiate optimizers dynamically.

## 3. How to Add a New Optimizer Implementation (Adapter)

To add a new optimization algorithm to the benchmark suite:

1.  **Create your Optimizer Adapter File**:
    Create a new Python file in this directory (e.g., `src/benchmark/optimizers/my_custom_optimizer.py`).

2.  **Implement Your Optimizer Class**:
    Define your optimizer class within this new file. It should inherit from one of the base optimizer classes provided in `src/benchmark/optimizer_protocols/`. Choose `NumpyBenchmarkOptimizer` if your optimizer works with NumPy arrays, `CupyBenchmarkOptimizer` for CuPy arrays, or `BenchmarkOptimizer` if you handle PyTorch tensors directly or need more custom control over `get_output_type()`.

    ```python
    # src/benchmark/optimizers/my_custom_optimizer.py
    import numpy as np
    from benchmark.evaluator import ModelEvaluator
    from benchmark.optimizer_protocols import NumpyBenchmarkOptimizer # Or CupyBenchmarkOptimizer

    class MyCustomOptimizer(NumpyBenchmarkOptimizer):
        def __init__(self, initial_params: np.ndarray, custom_param: float = 0.1, **config):
            super().__init__(initial_params, **config)
            self.custom_param = custom_param
            # self.params holds the flattened model parameters as a NumPy array

        def step(self, evaluator: ModelEvaluator) -> bool:
            # Get loss and gradients from the evaluator.
            # The data format (NumPy, CuPy) will match your chosen base class.
            loss, grad = evaluator.evaluate_with_grad() # 'grad' will be a NumPy array here

            # --- Your custom optimization logic goes here ---
            self.params = self.params - self.custom_param * grad
            # ------------------------------------------------

            # Update the model parameters in the evaluator
            evaluator.set_params(self.params)

            # Return True if the optimizer has converged and should stop, False otherwise
            return False
    ```

3.  **Register Your Optimizer**:
    Open `src/benchmark/optimizers/registry.py`.
    Import your new optimizer class and add it to the `OPTIMIZER_REGISTRY` dictionary.

    ```python
    # src/benchmark/optimizers/registry.py
    # ... other imports ...
    from .my_custom_optimizer import MyCustomOptimizer # Import your new optimizer

    OPTIMIZER_REGISTRY = {
        # ... existing optimizers ...
        "my-custom-optimizer": MyCustomOptimizer, # Map a unique string name to your class
    }
    ```
    Now your new optimizer can be selected by name when running benchmarks.
