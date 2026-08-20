# Evaluator DTOs Package

## 1. Overview

This package manages Data Transfer Objects (DTOs) used to abstract away the underlying tensor or array representations (e.g., PyTorch Tensors, NumPy ndarrays, CuPy ndarrays).

The core purpose of this system is to allow `ModelEvaluator` to provide data (parameters, gradients) in a format that any given optimizer can understand, and for the optimizer to return data in its native format, without the evaluator needing to know the specifics.

The key feature is a centralized conversion registry that allows for seamless, extensible, and decoupled conversions between different DTO types.

## 2. Architecture

-   **`evaluator_dto.py`**: Defines the `EvaluatorDto` abstract base class.
    -   Its `to(target_type)` method is the main entry point for conversions. It looks up and executes the appropriate converter from a central registry.

-   **`registry.py`**: Defines the `CONVERSION_REGISTRY`, a simple dictionary that maps `(SourceType, TargetType)` tuples to conversion functions. It also provides a `register_converter` helper function.

-   **`converters.py`**: This is the heart of the conversion logic.
    -   It imports all DTO types.
    -   It defines all the functions for converting between types (e.g., `pytorch_to_numpy`).
    -   It calls `registry.register_converter()` for each conversion function to populate the `CONVERSION_REGISTRY`.

-   **DTO modules** (`pytorch_tensor_evaluator_dto.py`, etc.): These files now only contain simple classes that inherit from `EvaluatorDto` and act as typed data holders. All conversion logic has been removed from them.

-   **`__init__.py`**: Exports all the DTO classes and, crucially, imports `converters.py` to ensure the registration logic runs as soon as the package is imported.

## 3. How to Add a New Evaluator DTO

Let's say you want to add support for JAX arrays.

1.  **Create the DTO file**:
    Create a new file `src/benchmark/evaluator_dtos/jax_array_dto.py`.

2.  **Define the DTO Class**:
    Inside the new file, define your class. It should be a simple data container.
    ```python
    # src/benchmark/evaluator_dtos/jax_array_dto.py
    import jax.numpy as jnp
    from .evaluator_dto import EvaluatorDto

    class JaxArrayDto(EvaluatorDto):
        def __init__(self, data: jnp.ndarray):
            self._data = data

        def data(self) -> jnp.ndarray:
            return self._data
    ```

3.  **Add Conversion Logic**:
    Open `src/benchmark/evaluator_dtos/converters.py`.

4.  **Import New DTO**:
    Import your new `JaxArrayDto` at the top of `converters.py`:
    ```python
    from .jax_array_dto import JaxArrayDto
    ```

5.  **Write Converter Functions**:
    Add the functions to convert to and from your new type. For example, to convert from PyTorch to JAX (if possible) and JAX to PyTorch.
    ```python
    # In converters.py

    # --- JAX -> Others ---
    def jax_to_pytorch(source_dto: JaxArrayDto, **params):
        # ... your logic to convert jax array to torch tensor ...
        # (This might involve converting JAX -> NumPy -> PyTorch)
        numpy_array = np.asarray(source_dto.data())
        device = params.get("device", "cpu")
        data = torch.from_numpy(numpy_array).float().to(device)
        return PyTorchTensorEvaluatorDto(data)

    # --- Others -> JAX ---
    def pytorch_to_jax(source_dto: PyTorchTensorEvaluatorDto, **params):
        numpy_array = source_dto.data().cpu().detach().numpy()
        data = jnp.asarray(numpy_array)
        return JaxArrayDto(data)
    ```

6.  **Register Converters**:
    At the bottom of `converters.py`, inside the `register_all()` function, add calls to `registry.register_converter()` for your new functions.
    ```python
    # In converters.py -> register_all()
    registry.register_converter(JaxArrayDto, PyTorchTensorEvaluatorDto, jax_to_pytorch)
    registry.register_converter(PyTorchTensorEvaluatorDto, JaxArrayDto, pytorch_to_jax)
    ```

7.  **Update `__init__.py`**:
    Finally, open `src/benchmark/evaluator_dtos/__init__.py` and add your new DTO class to the import list and the `__all__` list to make it easily accessible from other parts of the application.
    ```python
    # In __init__.py
    from .jax_array_dto import JaxArrayDto

    __all__ = [
        # ...,
        "JaxArrayDto",
    ]
    ```
