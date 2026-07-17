from __future__ import annotations

from typing import Any, Type, TypeVar

T = TypeVar("T", bound="EvaluatorDto")


class EvaluatorDto:
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def to(self, target_type: Type[T], **params) -> T:
        raise NotImplementedError(f"Conversion to {target_type.__name__} not available")

    def data(self) -> object:
        raise NotImplementedError("data() not implemented")
