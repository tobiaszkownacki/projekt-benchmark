from typing import override
from src.trainers.base_trainer import BaseTrainer
from src.config import BenchmarkConfig
from src.logging import Log
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class LbfgsTrainer(BaseTrainer):
    @override
    def train(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.TensorDataset,
        config: BenchmarkConfig,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        optimizer = self._get_optimizer()(model.parameters(), **self.params)
        criterion = config.criterion()
        log = Log(
            output_file=f"{self.__class__.__name__}-"
            f"{config.dataset_name}-"
            f"{self.name}-"
            f"{config.batch_size}.csv"
        )
        gradient_counter = 0

        while gradient_counter < config.gradient_counter_stop:
            model.train()
            model.to(device)
            running_loss = 0.0
            total = 0

            for inputs, targets in tqdm(train_loader, desc="Training"):
                inputs, targets = inputs.to(device), targets.to(device)
                if gradient_counter >= config.gradient_counter_stop:
                    break

                def closure():
                    optimizer.zero_grad()
                    loss, output = self._calculate_step(
                        model, inputs, targets, criterion, config.model_type
                    )
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
                log.add_number_of_samples(inputs.size(0))
                log.log(round(loss.item(), 4), config.save_interval)

                gradient_counter += 1

                with torch.no_grad():
                    output = model(inputs)
                    running_loss += loss.item()
                    _, predicted = torch.max(output, 1)
                    total += targets.size(0)

            print(f"Gradient counter: {gradient_counter}")
            train_loss = running_loss / total
            print(f"Train loss: {train_loss}")

    @override
    def _get_optimizer(
        self,
    ) -> callable:
        match self.name:
            case "lbfgs":
                return torch.optim.LBFGS
            case _:
                raise ValueError(f"Unsupported optimizer: {self.name}")

    @override
    def _get_optimizer_params(
        self,
        optimizer_params,
    ) -> dict:
        params = {
            "lr": optimizer_params.lr,
        }
        return params
