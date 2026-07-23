from benchmark.optimizers.cupy.cupy_adam import CupyAdam
from benchmark.optimizers.cupy.cupy_adamw import CupyAdamW
from benchmark.optimizers.cupy.cupy_lion import CupyLion
from benchmark.optimizers.cupy.cupy_rmsprop import CupyRMSProp
from benchmark.optimizers.cupy.cupy_sgd import CupySGD
from benchmark.optimizers.cupy.cupy_cmaes import CupyCMAES
from benchmark.optimizers.cupy.cupy_differential_evolution import (
    CupyDifferentialEvolution,
)
from benchmark.optimizers.cupy.cupy_des import CupyDES

from benchmark.optimizers.numpy.numpy_adam import NumpyAdam
from benchmark.optimizers.numpy.numpy_adamw import NumpyAdamW
from benchmark.optimizers.numpy.numpy_lion import NumpyLion
from benchmark.optimizers.numpy.numpy_rmsprop import NumpyRMSProp
from benchmark.optimizers.numpy.numpy_sgd import NumpySGD
from benchmark.optimizers.numpy.numpy_cmaes import NumpyCMAES
from benchmark.optimizers.numpy.numpy_differential_evolution import (
    NumpyDifferentialEvolution,
)
from benchmark.optimizers.numpy.numpy_des import NumpyDES



BUILTIN_OPTIMIZERS = {
    "adam": (CupyAdam, {"lr": 0.001}),
    "numpy_adam": (NumpyAdam, {"lr": 0.001}),

    "adamw": (CupyAdamW, {"lr": 0.001, "weight_decay": 0.01}),
    "numpy_adamw": (NumpyAdamW, {"lr": 0.001, "weight_decay": 0.01}),

    "lion": (CupyLion, {"lr": 1e-4, "weight_decay": 0.01}),
    "numpy_lion": (NumpyLion, {"lr": 1e-4, "weight_decay": 0.01}),

    "rmsprop": (CupyRMSProp, {"lr": 0.01, "alpha": 0.99}),
    "numpy_rmsprop": (NumpyRMSProp, {"lr": 0.01, "alpha": 0.99}),

    "sgd": (CupySGD, {"lr": 0.01}),
    "numpy_sgd": (NumpySGD, {"lr": 0.01}),

    "sgd_momentum": (CupySGD, {"lr": 0.01, "momentum": 0.9}),
    "numpy_sgd_momentum": (NumpySGD, {"lr": 0.01, "momentum": 0.9}),

    "cma-es": (CupyCMAES, {"sigma": 0.5}),
    "numpy_cma-es": (NumpyCMAES, {"sigma": 0.5}),

    "de": (CupyDifferentialEvolution, {"pop_size": 15, "F": 0.8, "CR": 0.7}),
    "numpy_de": (NumpyDifferentialEvolution, {"pop_size": 15, "F": 0.8, "CR": 0.7}),
    
    "des": (CupyDES, {"pop_size": 20, "sigma": 0.5}),
    "numpy_des": (NumpyDES, {"pop_size": 20, "sigma": 0.5}),
}
