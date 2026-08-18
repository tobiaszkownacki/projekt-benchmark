from benchmark.optimizers.registry import BUILTIN_OPTIMIZERS
from benchmark.optimizers.cupy.cupy_adam import CupyAdam
from benchmark.optimizers.cupy.cupy_adamw import CupyAdamW
from benchmark.optimizers.cupy.cupy_cmaes import CupyCMAES
from benchmark.optimizers.cupy.cupy_des import CupyDES
from benchmark.optimizers.cupy.cupy_differential_evolution import (
    CupyDifferentialEvolution,
)
from benchmark.optimizers.cupy.cupy_lion import CupyLion
from benchmark.optimizers.cupy.cupy_rmsprop import CupyRMSProp
from benchmark.optimizers.cupy.cupy_sgd import CupySGD

from benchmark.optimizers.numpy.numpy_adam import NumpyAdam
from benchmark.optimizers.numpy.numpy_adamw import NumpyAdamW
from benchmark.optimizers.numpy.numpy_cmaes import NumpyCMAES
from benchmark.optimizers.numpy.numpy_des import NumpyDES
from benchmark.optimizers.numpy.numpy_differential_evolution import (
    NumpyDifferentialEvolution,
)
from benchmark.optimizers.numpy.numpy_lion import NumpyLion
from benchmark.optimizers.numpy.numpy_rmsprop import NumpyRMSProp
from benchmark.optimizers.numpy.numpy_sgd import NumpySGD

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

