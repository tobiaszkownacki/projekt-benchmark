"""Import aliases that make ``src/benchmark_core`` importable.

The optimization engine lives in ``src/benchmark_core/optimization_engine/``
while its own absolute imports still name the previous layout
(``from benchmark.evaluator import ...``, ``from src.logging import ...``).
Until those imports are rewritten, the package cannot be imported at all.

This module binds the old names as synthetic packages whose ``__path__`` points
at the current directories, so the existing modules import unmodified. Binding a
``__path__`` rather than aliasing an already-imported module is what avoids the
cycle: ``optimization_engine/__init__.py`` imports through the legacy names
itself, so the alias has to resolve before that module is loaded.

Call :func:`install` before importing anything from the engine, and import
through the legacy names consistently -- mixing them with ``benchmark_core.*``
loads the same source twice under two identities.
"""

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
_ENGINE = _SRC / "benchmark_core" / "optimization_engine"

_installed = False


def _synthetic_package(name: str, path: Path) -> types.ModuleType:
    """A package object whose contents are the files in `path`."""
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _bind_module(alias: str, real_name: str) -> None:
    """Expose an existing module under a second, legacy name."""
    try:
        sys.modules[alias] = importlib.import_module(real_name)
    except ImportError:
        # Optional pieces (plotting needs matplotlib) may legitimately be
        # missing here; the caller finds out when it tries to use them.
        pass


def _install_cupy_stub() -> None:
    """Stand in for CuPy on a host that has no CUDA build of it.

    ``evaluator_dtos/__init__.py`` imports the CuPy DTO unconditionally and that
    DTO does ``import cupy`` at module scope, so the whole engine -- including
    every pure-NumPy optimizer -- is unimportable without a matching CuPy build.

    The stub supplies only the names touched during import: an ``ndarray``
    symbol used in annotations, and conversion entry points that raise when
    called, so a CuPy code path under the stub fails loudly rather than
    producing wrong numbers.
    """
    if "cupy" in sys.modules:
        return
    try:
        import cupy  # noqa: F401

        return
    except ImportError:
        pass

    stub = types.ModuleType("cupy")
    stub.__doc__ = "Compatibility stub installed by compat.benchmark_aliases."
    stub.IS_COMPAT_STUB = True

    class _Unavailable:
        """Placeholder for cupy.ndarray in type annotations."""

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "CuPy is not installed on this host. Use a NumPy optimizer, or run "
            "on a machine with a CuPy build matching its CUDA toolkit."
        )

    stub.ndarray = _Unavailable
    stub.asarray = _unavailable
    stub.asnumpy = _unavailable
    stub.from_dlpack = _unavailable
    stub.array = _unavailable
    sys.modules["cupy"] = stub


def install(cupy_stub: bool = True) -> None:
    """Register the legacy package names. Idempotent.

    cupy_stub: install a stub CuPy when the real one is absent, so that
    NumPy-only work is possible on a CPU host. Pass False to require the real
    library.
    """
    global _installed
    if _installed:
        return

    if cupy_stub:
        _install_cupy_stub()

    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    # `benchmark` was the engine package, `src` the repository root treated as
    # one. Neither directory exists under those names any more.
    if "benchmark" not in sys.modules:
        _synthetic_package("benchmark", _ENGINE)
    if "src" not in sys.modules:
        _synthetic_package("src", _SRC)

    sys.modules.setdefault("src.benchmark", sys.modules["benchmark"])

    # Single modules that moved rather than whole packages.
    _bind_module("src.logging", "benchmark_core.custom_logging")
    _bind_module("src.plotting", "benchmark_core.plotting")

    _installed = True


def engine():
    """Convenience accessor: install(), then hand back the usable pieces."""
    install()
    from benchmark.evaluator import ModelEvaluator  # noqa: E402
    from benchmark.evaluator_dtos import (  # noqa: E402
        NumpyNdarrayTensorEvaluatorDto,
        PyTorchTensorEvaluatorDto,
    )

    return {
        "ModelEvaluator": ModelEvaluator,
        "PyTorchTensorEvaluatorDto": PyTorchTensorEvaluatorDto,
        "NumpyNdarrayTensorEvaluatorDto": NumpyNdarrayTensorEvaluatorDto,
    }
