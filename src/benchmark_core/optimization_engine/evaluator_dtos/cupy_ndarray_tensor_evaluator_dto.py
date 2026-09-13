from __future__ import annotations

from typing import TYPE_CHECKING

from .evaluator_dto import EvaluatorDto

if TYPE_CHECKING:
    # The DTO only carries a reference to an array somebody else allocated, so
    # CuPy is needed to PRODUCE one, not to import this module. Importing it
    # eagerly made every pure-NumPy optimizer unloadable on a host without a
    # matching CuPy build.
    import cupy as cp


class CupyNdarrayTensorEvaluatorDto(EvaluatorDto):
    def __init__(self, data: cp.ndarray):
        self._data = data

    def data(self) -> cp.ndarray:
        return self._data
