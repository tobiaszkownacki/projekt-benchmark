from __future__ import annotations
from typing import Any
import torch
from typing import Type
from typing import TypeVar
from torch import Tensor
from torch.utils.dlpack import to_dlpack
import numpy as np
import cupy as cp

T = TypeVar("T", bound="EvaluatorDto")


class EvaluatorDto:
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def to(self, target_type: Type[T], **params) -> T:
        raise NotImplementedError(f"Conversion to {target_type.__name__} not available")


class PyTorchTensorEvaluatorDto(EvaluatorDto):
    def __init__(self, data: Tensor):
        self.data = data

    def to(self, target_type: Type[T], **params) -> T:
        match target_type.__name__:
            case NumpyNdarrayTensorEvaluatorDto.__name__:
                return target_type(self.data.cpu().numpy().flatten())
            case CupyNdarrayTensorEvaluatorDto.__name__:
                dlpack_data = to_dlpack(self.data)
                return target_type(cupy.from_dlpack(dlpack_data))
            case _:
                return super().to(target_type)


class CupyNdarrayTensorEvaluatorDto(EvaluatorDto):
    def __init__(self, data: cp.ndarray):
        self.data = data

    def to(self, target_type: Type[T], **params) -> T:
        match target_type.__name__:
            case PyTorchTensorEvaluatorDto.__name__:
                dlpack_data = self.data.to_dlpack()
                return target_type(torch.from_dlpack(dlpack_data))
            case _:
                return super().to(target_type)


class NumpyNdarrayTensorEvaluatorDto(EvaluatorDto):
    def __init__(self, data: np.ndarray):
        self.data = data

    def to(self, target_type: Type[T], **params) -> T:
        match target_type.__name__:
            case PyTorchTensorEvaluatorDto.__name__:
                return target_type(torch.from_numpy(self.data))
            case _:
                return super().to(target_type)
