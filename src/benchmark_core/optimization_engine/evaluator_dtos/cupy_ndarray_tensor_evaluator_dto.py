import cupy as cp
from .evaluator_dto import EvaluatorDto


class CupyNdarrayTensorEvaluatorDto(EvaluatorDto):
    def __init__(self, data: cp.ndarray):
        self._data = data

    def data(self) -> cp.ndarray:
        return self._data
