"""
A central registry for DTO conversion functions.
"""

CONVERSION_REGISTRY = {}


def register_converter(source_type, target_type, func):
    """
    Registers a function to convert from one DTO type to another.

    Args:
        source_type: The class of the source DTO.
        target_type: The class of the target DTO.
        func: A function that takes a source DTO instance and returns a target DTO instance.
              Signature: converter(source_dto, **params) -> target_dto
    """
    if (source_type, target_type) in CONVERSION_REGISTRY:
        # This could be a warning instead of an error if overwriting is desired.
        raise ValueError(
            f"Converter from {source_type.__name__} to {target_type.__name__} already registered."
        )
    CONVERSION_REGISTRY[(source_type, target_type)] = func
