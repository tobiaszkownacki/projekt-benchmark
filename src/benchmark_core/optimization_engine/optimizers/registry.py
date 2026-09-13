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

BUILTIN_OPTIMIZERS = {
    "numpy_adam": (NumpyAdam, {"lr": 0.001}),
    "numpy_adamw": (NumpyAdamW, {"lr": 0.001, "weight_decay": 0.01}),
    "numpy_lion": (NumpyLion, {"lr": 1e-4, "weight_decay": 0.01}),
    "numpy_rmsprop": (NumpyRMSProp, {"lr": 0.001, "alpha": 0.99}),
    "numpy_sgd": (NumpySGD, {"lr": 0.01}),
    "numpy_sgd_momentum": (NumpySGD, {"lr": 0.01, "momentum": 0.9}),
    "numpy_cma-es": (NumpyCMAES, {"sigma": 0.1}),
    "numpy_de": (NumpyDifferentialEvolution, {"pop_size": 50, "F": 0.8, "CR": 0.7}),
    "numpy_des": (NumpyDES, {"pop_size": 20, "sigma": 0.5}),
}

# Every CuPy optimizer needs a CuPy build matching the host's CUDA toolkit. On a
# host without one the NumPy entries above are the whole registry: the
# unprefixed names stay absent rather than resolving to a different backend,
# because which backend produced a number is part of the measurement.
try:
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
except ImportError:
    pass
else:
    BUILTIN_OPTIMIZERS.update(
        {
            "adam": (CupyAdam, {"lr": 0.001}),
            "adamw": (CupyAdamW, {"lr": 0.001, "weight_decay": 0.01}),
            "lion": (CupyLion, {"lr": 1e-4, "weight_decay": 0.01}),
            "rmsprop": (CupyRMSProp, {"lr": 0.001, "alpha": 0.99}),
            "sgd": (CupySGD, {"lr": 0.01}),
            "sgd_momentum": (CupySGD, {"lr": 0.01, "momentum": 0.9}),
            "cma-es": (CupyCMAES, {"sigma": 0.1}),
            "de": (CupyDifferentialEvolution, {"pop_size": 50, "F": 0.8, "CR": 0.7}),
            "des": (CupyDES, {"pop_size": 20, "sigma": 0.5}),
        }
    )
