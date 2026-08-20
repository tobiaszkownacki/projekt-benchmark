"""Import aliases that make ``src/benchmark_core`` importable again.

The refactor in 891bf11..a9b0c6c moved the optimization engine from
``src/benchmark/`` to ``src/benchmark_core/optimization_engine/`` but left every
absolute import inside it pointing at the old layout. Roughly forty modules
still say ``from benchmark.evaluator import ModelEvaluator`` or
``from src.logging import Log``, so the package does not import at all -- the
evaluator, the optimizers, the runner and the protocol validator are all present
on main and all unreachable.

The real fix is to rewrite those imports, and it should be done. It is not done
here: the optimization engine belongs to the project lead (§4 -- "wejście w cudzy
moduł wymaga uzgodnienia"), and a forty-file rename landed from a web branch
would collide with whatever is in flight there. The defect is reported instead.

Meanwhile this module re-creates the old package names as views onto the
directories that now hold the code, so the existing modules import unmodified.

The names are bound as synthetic packages with a ``__path__`` rather than as
aliases of already-imported modules, and that detail is what makes it work:
``optimization_engine/__init__.py`` itself does ``from benchmark.runner import
...``, so importing it in order to alias it needs the alias to exist first.
Pointing ``benchmark.__path__`` straight at the directory lets ``benchmark.runner``
resolve as an ordinary submodule and the cycle never forms.

Call install() before importing anything from the engine, and import through the
legacy names consistently -- mixing them with ``benchmark_core.*`` would load the
same source twice under two identities. Delete this module once the imports
upstream are fixed; it is a bridge with a known expiry, not a design.
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

    ``evaluator_dtos/__init__.py`` imports the CuPy DTO unconditionally, and that
    DTO does ``import cupy`` at module scope. The effect is that the entire
    optimization engine -- including every pure-NumPy optimizer -- is unimportable
    on any machine without a matching CuPy build, which is a second defect worth
    reporting separately from the stale import paths.

    The stub supplies only the names touched during import: an ``ndarray``
    symbol used in annotations, and conversion entry points that raise if they
    are ever actually called. Nothing silently produces wrong numbers -- a CuPy
    code path under the stub fails loudly instead of pretending to work.
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

    # `benchmark` was the engine package; `src` was the repository root treated
    # as a package. Neither exists on main any more.
    if "benchmark" not in sys.modules:
        _synthetic_package("benchmark", _ENGINE)
    if "src" not in sys.modules:
        _synthetic_package("src", _SRC)

    sys.modules.setdefault("src.benchmark", sys.modules["benchmark"])

    # Single modules that moved rather than whole packages.
    _bind_module("src.logging", "benchmark_core.logging")
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
