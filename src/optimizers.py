from torch.optim import Adam, SGD, AdamW, RMSprop
from torch.optim import Optimizer


class OptimizerFactory:
    @staticmethod
    def get_optimizer(optimizer_name: str) -> Optimizer:
        match optimizer_name:
            case 'adam':
                return Adam()
            case 'adamw':
                return AdamW()
            case 'sgd':
                return SGD()
            case 'rmsprop':
                return RMSprop()
            case _:
                raise ValueError(f'Unsupported optimizer: {optimizer_name}')
