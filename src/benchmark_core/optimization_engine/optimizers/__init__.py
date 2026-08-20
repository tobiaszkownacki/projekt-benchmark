from src.benchmark_core.optimization_engine.optimizers.registry import BUILTIN_OPTIMIZERS
from src.benchmark_core.optimization_engine.optimizers.cupy.cupy_adam import CupyAdam
from src.benchmark_core.optimization_engine.optimizers.cupy.cupy_adamw import CupyAdamW
from src.benchmark_core.optimization_engine.optimizers.cupy.cupy_cmaes import CupyCMAES
from src.benchmark_core.optimization_engine.optimizers.cupy.cupy_des import CupyDES
from src.benchmark_core.optimization_engine.optimizers.cupy.cupy_differential_evolution import (
    CupyDifferentialEvolution,
)
from src.benchmark_core.optimization_engine.optimizers.cupy.cupy_lion import CupyLion
from src.benchmark_core.optimization_engine.optimizers.cupy.cupy_rmsprop import CupyRMSProp
from src.benchmark_core.optimization_engine.optimizers.cupy.cupy_sgd import CupySGD

from src.benchmark_core.optimization_engine.optimizers.numpy.numpy_adam import NumpyAdam
from src.benchmark_core.optimization_engine.optimizers.numpy.numpy_adamw import NumpyAdamW
from src.benchmark_core.optimization_engine.optimizers.numpy.numpy_cmaes import NumpyCMAES
from src.benchmark_core.optimization_engine.optimizers.numpy.numpy_des import NumpyDES
from src.benchmark_core.optimization_engine.optimizers.numpy.numpy_differential_evolution import (
    NumpyDifferentialEvolution,
)
from src.benchmark_core.optimization_engine.optimizers.numpy.numpy_lion import NumpyLion
from src.benchmark_core.optimization_engine.optimizers.numpy.numpy_rmsprop import NumpyRMSProp
from src.benchmark_core.optimization_engine.optimizers.numpy.numpy_sgd import NumpySGD

__all__ = [
    "PyTorchOptimizerAdapter",
    "CupyAdam",
    "CupyAdamW",
    "CupyLion",
    "CupyRMSProp",
    "CupySGD",
    "CupyCMAES",
    "CupyDifferentialEvolution",
    "CupyDES",
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

