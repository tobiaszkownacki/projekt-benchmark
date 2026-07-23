"""
This file defines and registers all DTO conversion functions.
Importing this file will populate the conversion registry.
"""

import cupy as cp
import torch

from . import registry
from .cupy_ndarray_tensor_evaluator_dto import CupyNdarrayTensorEvaluatorDto
from .numpy_ndarray_tensor_evaluator_dto import NumpyNdarrayTensorEvaluatorDto
from .pytorch_tensor_evaluator_dto import PyTorchTensorEvaluatorDto

# --- PyTorch -> Others ---


def pytorch_to_numpy(source_dto: PyTorchTensorEvaluatorDto, **params):
    """Converts a PyTorch DTO to a NumPy DTO."""
    data = source_dto.data().cpu().detach().numpy().flatten()
    return NumpyNdarrayTensorEvaluatorDto(data)


def pytorch_to_cupy(source_dto: PyTorchTensorEvaluatorDto, **params):
    """Converts a PyTorch DTO to a CuPy DTO."""
    dlpack = torch.to_dlpack(source_dto.data())
    data = cp.from_dlpack(dlpack)
    return CupyNdarrayTensorEvaluatorDto(data)


# --- NumPy -> Others ---


def numpy_to_pytorch(source_dto: NumpyNdarrayTensorEvaluatorDto, **params):
    """Converts a NumPy DTO to a PyTorch DTO."""
    device = params.get("device", "cpu")
    data = torch.from_numpy(source_dto.data()).float().to(device)
    return PyTorchTensorEvaluatorDto(data)


# --- CuPy -> Others ---


def cupy_to_pytorch(source_dto: CupyNdarrayTensorEvaluatorDto, **params):
    """Converts a CuPy DTO to a PyTorch DTO."""
    dlpack = source_dto.data().toDlpack()
    data = torch.from_dlpack(dlpack)
    return PyTorchTensorEvaluatorDto(data)


# =================================================================
# Register all conversion functions
# =================================================================
def register_all():
    registry.register_converter(
        PyTorchTensorEvaluatorDto, NumpyNdarrayTensorEvaluatorDto, pytorch_to_numpy
    )
    registry.register_converter(
        PyTorchTensorEvaluatorDto, CupyNdarrayTensorEvaluatorDto, pytorch_to_cupy
    )
    registry.register_converter(
        NumpyNdarrayTensorEvaluatorDto, PyTorchTensorEvaluatorDto, numpy_to_pytorch
    )
    registry.register_converter(
        CupyNdarrayTensorEvaluatorDto, PyTorchTensorEvaluatorDto, cupy_to_pytorch
    )


register_all()
