from src.benchmark.optimizers.adam_adapter import AdamAdapter
from src.benchmark.optimizers.adamw_adapter import AdamWAdapter
from src.benchmark.optimizers.lion_adapter import LionAdapter
from src.benchmark.optimizers.rmsprop_adapter import RMSPropAdapter
from src.benchmark.optimizers.sgd_adapter import SGDAdapter
from src.benchmark.optimizers.cmaes_adapter import CMAESAdapter
from src.benchmark.optimizers.differential_evolution_adapter import (
    DifferentialEvolutionAdapter,
)
from src.benchmark.optimizers.des_adapter import DESAdapter


BUILTIN_OPTIMIZERS = {
    "adam": (AdamAdapter, {"lr": 0.001}),
    "adamw": (AdamWAdapter, {"lr": 0.001, "weight_decay": 0.01}),
    "lion": (LionAdapter, {"lr": 1e-4, "weight_decay": 0.01}),
    "rmsprop": (RMSPropAdapter, {"lr": 0.01, "alpha": 0.99}),
    "sgd": (SGDAdapter, {"lr": 0.01}),
    "sgd_momentum": (SGDAdapter, {"lr": 0.01, "momentum": 0.9}),
    "cma-es": (CMAESAdapter, {"sigma": 0.5}),
    "de": (DifferentialEvolutionAdapter, {"pop_size": 15, "F": 0.8, "CR": 0.7}),
    "des": (DESAdapter, {"pop_size": 20, "sigma": 0.5}),
}
