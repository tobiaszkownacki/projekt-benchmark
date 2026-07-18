from __future__ import annotations

from typing import Any, Type, TypeVar

T = TypeVar("T", bound="EvaluatorDto")


class EvaluatorDto:
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def to(self, target_type: Type[T], **params) -> T:
        from . import registry  # Local import to prevent circular dependencies

        # Direct conversion if types are the same
        if isinstance(self, target_type):
            return self

        converter_func = registry.CONVERSION_REGISTRY.get((type(self), target_type))
        if converter_func:
            return converter_func(self, **params)

        raise NotImplementedError(
            f"Conversion from {type(self).__name__} to {target_type.__name__} is not registered."
        )

    def data(self) -> object:
        raise NotImplementedError("data() not implemented")
