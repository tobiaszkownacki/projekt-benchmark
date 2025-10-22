from torch import nn


class CriterionFactory:
    @staticmethod
    def get_criterion(criterion_name: str) -> nn.Module:
        match criterion_name:
            case "cross_entropy":
                return nn.CrossEntropyLoss()
            case "mse_loss":
                return nn.MSELoss()
            case "mae_loss":
                return nn.L1Loss()
            case _:
                raise ValueError(f"Unsupported criterion: {criterion_name}")
