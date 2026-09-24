from benchmark_core.optimization_engine.optimizers.numpy.numpy_adam import NumpyAdam
from benchmark_core.optimization_engine.optimizers.numpy.numpy_adamw import NumpyAdamW
from benchmark_core.optimization_engine.optimizers.numpy.numpy_cmaes import NumpyCMAES
from benchmark_core.optimization_engine.optimizers.numpy.numpy_des import NumpyDES
from benchmark_core.optimization_engine.optimizers.numpy.numpy_differential_evolution import (
    NumpyDifferentialEvolution,
)
from benchmark_core.optimization_engine.optimizers.numpy.numpy_lion import NumpyLion
from benchmark_core.optimization_engine.optimizers.numpy.numpy_rmsprop import NumpyRMSProp
from benchmark_core.optimization_engine.optimizers.numpy.numpy_sgd import NumpySGD
from benchmark_core.optimization_engine.optimizers.registry import BUILTIN_OPTIMIZERS

__all__ = [
    "NumpyAdam",
    "NumpyAdamW",
    "NumpyLion",
    "NumpyRMSProp",
    "NumpySGD",
    "NumpyCMAES",
    "NumpyDifferentialEvolution",
    "NumpyDES",
    "BUILTIN_OPTIMIZERS",
]

# The CuPy optimizers are exported only where CuPy actually loads -- see the
# note in registry.py.
try:
    from benchmark_core.optimization_engine.optimizers.cupy.cupy_adam import CupyAdam
    from benchmark_core.optimization_engine.optimizers.cupy.cupy_adamw import CupyAdamW
    from benchmark_core.optimization_engine.optimizers.cupy.cupy_cmaes import CupyCMAES
    from benchmark_core.optimization_engine.optimizers.cupy.cupy_des import CupyDES
    from benchmark_core.optimization_engine.optimizers.cupy.cupy_differential_evolution import (
        CupyDifferentialEvolution,
    )
    from benchmark_core.optimization_engine.optimizers.cupy.cupy_lion import CupyLion
    from benchmark_core.optimization_engine.optimizers.cupy.cupy_rmsprop import CupyRMSProp
    from benchmark_core.optimization_engine.optimizers.cupy.cupy_sgd import CupySGD
except ImportError:
    pass
else:
    __all__ += [
        "CupyAdam",
        "CupyAdamW",
        "CupyLion",
        "CupyRMSProp",
        "CupySGD",
        "CupyCMAES",
        "CupyDifferentialEvolution",
        "CupyDES",
    ]
