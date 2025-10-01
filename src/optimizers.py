from torch.optim import Adam, SGD, AdamW, RMSprop, LBFGS
from torch.optim import Optimizer
from cma import CMAEvolutionStrategy


class OptimizerFactory:
    @staticmethod
    def get_optimizer(optimizer_name: str) -> Optimizer:
        match optimizer_name:
            case "adam":
                return Adam
            case "adamw":
                return AdamW
            case "sgd":
                return SGD
            case "rmsprop":
                return RMSprop
            case "lbfgs":
                return LBFGS
            case "cma-es":
                return CMAEvolutionStrategy
            case _:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")
