from torch import Tensor
from .evaluator_dto import EvaluatorDto


class PyTorchTensorEvaluatorDto(EvaluatorDto):
    def __init__(self, data: Tensor):
        self._data = data

    def data(self) -> Tensor:
        return self._data
