from benchmark.evaluator_dtos import PyTorchTensorEvaluatorDto
from benchmark.evaluator_dtos.evaluator_dto import T
from benchmark.evaluator_dtos import EvaluatorDto
from typing import Type

import numpy as np
import torch


class NumpyNdarrayTensorEvaluatorDto(EvaluatorDto):
    def __init__(self, data: np.ndarray):
        self._data = data

    def to(self, target_type: Type[T], **params) -> T:
        if target_type is PyTorchTensorEvaluatorDto:
            device = params.get("device", "cpu")
            return target_type(torch.from_numpy(self.data()).float().to(device))
        else:
            return super().to(target_type)

    def data(self) -> np.ndarray:
        return self._data
