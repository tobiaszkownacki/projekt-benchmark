from benchmark.evaluator_dtos.evaluator_dto import T
from typing import Type

import cupy as cp
from torch import Tensor, to_dlpack

from benchmark.evaluator_dtos import (
    CupyNdarrayTensorEvaluatorDto,
    EvaluatorDto,
    NumpyNdarrayTensorEvaluatorDto,
)


class PyTorchTensorEvaluatorDto(EvaluatorDto):
    def __init__(self, data: Tensor):
        self._data = data

    def to(self, target_type: Type[T], **params) -> T:
        if target_type is NumpyNdarrayTensorEvaluatorDto:
            return target_type(self.data().cpu().detach().numpy().flatten())
        elif target_type is CupyNdarrayTensorEvaluatorDto:
            dlpack_data = to_dlpack(self.data())
            return target_type(cp.from_dlpack(dlpack_data))
        else:
            return super().to(target_type)

    def data(self) -> Tensor:
        return self._data
