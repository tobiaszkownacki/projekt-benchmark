from typing import override

import adopt
from src.trainers.base_trainer import BaseTrainer
from src.config import BenchmarkConfig, OptimizerParams, SchedulerConfig
from src.logging import Log
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from src.optimizers.lion_optimizers.lion_pytorch import Lion
from src.optimizers.gravity_optimizers.gravity_pytorch import Gravity

class GradientTrainer(BaseTrainer):
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
        criterion = CrossEntropyLoss()
        optimizer = self._get_optimizer()(
            model.parameters(),
            **self.params,
        )
        scheduler = self._get_scheduler(config.scheduler_config, optimizer)

        log = Log(
            output_file=f"{self.__class__.__name__}-"
            f"{config.dataset_name}-"
            f"{self.name}-"
            f"{config.batch_size}-"
            f"{config.scheduler_config.name}.csv"
        )

        gradient_counter = 0
        train_losses = []
        train_accuracies = []

        while gradient_counter < config.gradient_counter_stop:
            model.train()
            model.to(device)
            running_loss = 0.0
            correct = 0
            total = 0
            batch_count = 0

            for inputs, targets in tqdm(train_loader, desc="Training"):
                inputs, targets = inputs.to(device), targets.to(device)
                if gradient_counter >= config.gradient_counter_stop:
                    break

                optimizer.zero_grad()

                output = model(inputs)
                loss = criterion(output, targets)
                gradient_counter += 1
                loss = loss.to(device)
                loss.backward()
                optimizer.step()

                current_lr = optimizer.param_groups[0]["lr"]

                log.add_number_of_samples(targets.size(0))
                log.log(
                    round(loss.item(), 4), config.save_interval, round(current_lr, 8)
                )

                running_loss += loss.item()
                batch_count += 1

                _, predicted = torch.max(output, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            print(f"Gradient counter: {gradient_counter}")
            if batch_count > 0:
                train_loss = running_loss / batch_count
                train_losses.append(train_loss)
                train_accuracy = 100 * correct / total
                train_accuracies.append(train_accuracy)

                if scheduler is not None:
                    if config.scheduler_config.name == "reduceonplateau":
                        scheduler.step(train_loss)
                    else:
                        scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Train loss: {train_loss}, Train accuracy: {train_accuracy}")
                print(f"Learning rate: {current_lr:.8f}")
            else:
                print("No samples processes in this epoch")

    def _validate(self, model, val_loader, criterion, device):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        return avg_val_loss, val_accuracy

    @override
    def _get_optimizer(self) -> callable:
        match self.name:
            case "adam":
                return torch.optim.Adam
            case "adamw":
                return torch.optim.AdamW
            case "sgd":
                return torch.optim.SGD
            case "lion":
                return Lion
            case "rmsprop":
                return torch.optim.RMSprop
            case "adopt":
                return adopt.ADOPT
            case "gravity":
                return Gravity
            case _:
                raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    @override
    def _get_optimizer_params(self, optimizer_params: OptimizerParams) -> dict:
        params = {
            "lr": optimizer_params.lr,
            "weight_decay": optimizer_params.weight_decay,
        }
        match self.name:
            case "sgd":
                params.update(
                    {
                        "momentum": optimizer_params.momentum,
                        "dampening": optimizer_params.dampening,
                        "maximize": optimizer_params.maximize,
                    }
                )
            case "adam" | "adamw" | "adopt":
                params.update(
                    {
                        "betas": optimizer_params.betas,
                        "eps": optimizer_params.eps,
                        "maximize": optimizer_params.maximize,
                    }
                )
            case "rmsprop":
                params.update(
                    {
                        "momentum": optimizer_params.momentum,
                        "alpha": optimizer_params.alpha,
                        "centered": optimizer_params.centered,
                        "maximize": optimizer_params.maximize,
                    }
                )
            case "lion":
                params.update(
                    {
                        "betas": optimizer_params.betas,
                        "weight_decay": optimizer_params.weight_decay,
                    }
                )
        print(params)
        return params

    def _get_scheduler(
        self, scheduler_config: SchedulerConfig, optimizer: torch.optim
    ) -> callable:
        match scheduler_config.name:
            case "none":
                return None
            case "steplr":
                return torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_config.step_size,
                    gamma=scheduler_config.gamma,
                )
            case "exponentiallr":
                return torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=scheduler_config.gamma
                )
            case "reduceonplateau":
                return torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=scheduler_config.gamma,
                    patience=scheduler_config.patience,
                )
            case "cosineannealinglr":
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=scheduler_config.step_size
                )
            case _:
                raise ValueError(
                    f"Unsupported scheduler: {scheduler_config.scheduler_name}"
                )
