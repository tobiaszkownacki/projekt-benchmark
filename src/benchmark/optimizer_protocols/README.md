# Optimizer Protocols Package

## 1. Overview

This package defines the interfaces (protocols) and optional base classes for all optimization algorithms that can be used within the benchmark suite. It ensures that any optimizer, whether it's gradient-based or gradient-free, can communicate with the `ModelEvaluator` in a standardized way.

The core idea is to decouple the optimizers from the `ModelEvaluator`'s internal data representation (which is PyTorch). This is achieved by having optimizers declare what data format they expect via `EvaluatorDto` types.

## 2. Architecture

-   **`benchmarkable_optimizer.py`**: Defines the `BenchmarkableOptimizer` Protocol.
    -   This is the most critical file. Any class that implements the `step(self, evaluator: ModelEvaluator) -> bool` method implicitly satisfies this protocol.
    -   Your optimizer does **not** need to inherit from anything in this package, as long as it has a compliant `step` method and a `get_output_type` class method.

-   **`benchmark_optimizer.py`**: Defines an optional `BenchmarkOptimizer` base class.
    -   It provides a standard `__init__` that stores initial parameters and a configuration dictionary.
    -   It requires subclasses to implement `step()` and `get_output_type()`.

-   **`numpy_benchmark_optimizer.py` & `cupy_benchmark_optimizer.py`**:
    -   These are specialized base classes inheriting from `BenchmarkOptimizer`.
    -   They are convenience classes for optimizers that work with NumPy or CuPy arrays.
    -   Their main purpose is to pre-define the `get_output_type()` method, which tells the `ModelEvaluator` to provide data in the form of `NumpyNdarrayTensorEvaluatorDto` or `CupyNdarrayTensorEvaluatorDto`.

## 3. How to Add a New Optimizer

There are two main ways to add a new optimizer.

### A) Inheriting from a Base Class (Recommended)

This is the easiest approach, especially for NumPy/CuPy based optimizers.

1.  **Create your optimizer file**:
    For example, `src/benchmark/optimizers/my_cool_optimizer.py`.

2.  **Inherit and Implement**:
    Choose the appropriate base class and implement the `step` method.
    ```python
    # src/benchmark/optimizers/my_cool_optimizer.py
    from benchmark.optimizer_protocols import NumpyBenchmarkOptimizer
    from benchmark.evaluator import ModelEvaluator

    class MyCoolOptimizer(NumpyBenchmarkOptimizer):
        def __init__(self, initial_params, my_arg=0.5, **config):
            super().__init__(initial_params, **config)
            self.my_arg = my_arg
            # self.params contains the initial parameters as a NumPy array

        def step(self, evaluator: ModelEvaluator) -> bool:
            # The evaluator will give you params and gradients as NumPy arrays
            # because the base class specified NumpyNdarrayTensorEvaluatorDto.
            loss, grad = evaluator.evaluate_with_grad() # grad is a NumPy array

            # ... your optimization logic ...
            self.params = self.params - self.my_arg * grad

            # Give the updated parameters back to the evaluator
            evaluator.set_params(self.params)

            # Return False to continue, True to stop
            return False
    ```

### B) Implementing the Protocol from Scratch

If your optimizer has a very unique structure, you can implement the protocol without any inheritance.

```python
from benchmark.evaluator import ModelEvaluator
from benchmark.evaluator_dtos import MyCustomDto # Assuming you created this

class MyStandaloneOptimizer:
    # No inheritance needed

    @classmethod
    def get_output_type(cls):
        # You MUST implement this to tell the evaluator what data you want
        return MyCustomDto

    def __init__(self, initial_params, **config):
        # self.params now holds data wrapped in MyCustomDto
        self.params = initial_params
        # ...

    def step(self, evaluator: ModelEvaluator) -> bool:
        # This method signature makes the class compliant with the protocol
        # ... your logic ...
        return False
```

## 4. How to Add a New Optimizer Base Class

If you are creating a new family of optimizers that use a different data backend (e.g., JAX), you can create a new base class for convenience.

1.  **Ensure the DTO Exists**:
    First, follow the guide in `src/benchmark/evaluator_dtos/README.md` to create your `JaxArrayDto` and register its converters.

2.  **Create the Base Class File**:
    Create `src/benchmark/optimizer_protocols/jax_benchmark_optimizer.py`.

3.  **Define the Base Class**:
    ```python
    # src/benchmark/optimizer_protocols/jax_benchmark_optimizer.py
    from typing import Type
    from .benchmark_optimizer import BenchmarkOptimizer
    from benchmark.evaluator_dtos import JaxArrayDto, EvaluatorDto

    class JaxBenchmarkOptimizer(BenchmarkOptimizer):
        @classmethod
        def get_output_type(cls) -> Type[EvaluatorDto]:
            # This is the key step: associate this base class
            # with a specific DTO type.
            return JaxArrayDto
    ```
    Note: You only need to implement `get_output_type`. The `step` method is intentionally left unimplemented to force the final optimizer to provide its own logic.

4.  **Update `__init__.py`**:
    Add your new `JaxBenchmarkOptimizer` to the `__all__` list in `src/benchmark/optimizer_protocols/__init__.py` to make it easily importable.
