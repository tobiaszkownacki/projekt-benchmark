import numpy as np
from .evaluator_dto import EvaluatorDto


class NumpyNdarrayTensorEvaluatorDto(EvaluatorDto):
    def __init__(self, data: np.ndarray):
        self._data = data

    def data(self) -> np.ndarray:
        return self._data
