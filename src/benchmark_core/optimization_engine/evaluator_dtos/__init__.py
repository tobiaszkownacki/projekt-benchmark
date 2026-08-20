from src.benchmark_core.optimization_engine.evaluator_dtos.cupy_ndarray_tensor_evaluator_dto import (
    CupyNdarrayTensorEvaluatorDto,
)
from src.benchmark_core.optimization_engine.evaluator_dtos.evaluator_dto import EvaluatorDto
from src.benchmark_core.optimization_engine.evaluator_dtos.numpy_ndarray_tensor_evaluator_dto import (
    NumpyNdarrayTensorEvaluatorDto,
)
from src.benchmark_core.optimization_engine.evaluator_dtos.pytorch_tensor_evaluator_dto import (
    PyTorchTensorEvaluatorDto,
)

__all__ = [
    "CupyNdarrayTensorEvaluatorDto",
    "EvaluatorDto",
    "NumpyNdarrayTensorEvaluatorDto",
    "PyTorchTensorEvaluatorDto",
]

# Import converters to trigger registration
from . import converters
