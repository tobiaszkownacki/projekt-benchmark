from benchmark.evaluator_dtos import PyTorchTensorEvaluatorDto
from benchmark.evaluator_dtos.evaluator_dto import T
from benchmark.evaluator_dtos import EvaluatorDto
from typing import Type

import cupy as cp
import torch


class CupyNdarrayTensorEvaluatorDto(EvaluatorDto):
    def __init__(self, data: cp.ndarray):
        self._data = data

    def to(self, target_type: Type[T], **params) -> T:
        if target_type is PyTorchTensorEvaluatorDto:
            dlpack_data = self.data().toDlpack()
            return target_type(torch.from_dlpack(dlpack_data))
        else:
            return super().to(target_type)

    def data(self) -> cp.ndarray:
        return self._data
