from src.trainers.base_trainer import BaseTrainer
from src.config import BenchmarkConfig, SchedulerConfig
from src.logging import Log
from src.plots import ModelAnalyzer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import CrossEntropyLoss


class GradientTrainer(BaseTrainer):
    def train(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.TensorDataset,
        config: BenchmarkConfig,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.initialization_xavier:
            analyzer = ModelAnalyzer()
            initial_weights = analyzer.capture_initial_weights(model, "xavier_uniform")

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        criterion = config.criterion()
        optimizer = self._get_optimizer(config.optimizer_config.optimizer_name)(
            model.parameters(),
            lr=config.gradient_optimizer_params.learning_rate,
            weight_decay=config.gradient_optimizer_params.weight_decay,
        )
        scheduler = self._get_scheduler(config.scheduler_config, optimizer)

        log = Log(
            output_file=f"{self.__class__.__name__}-"
            f"{config.dataset_name}-"
            f"{config.optimizer_config.optimizer_name}-"
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

                log.increment_number_of_samples(targets.size(0))
                log.increment_mini_batches(1)
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
                print(f"Train loss: {train_loss}")
                print(f"Learning rate: {current_lr:.8f}")
            else:
                print("No samples processes in this epoch")

        if config.initialization_xavier:
            if train_losses and train_accuracies:
                final_loss = train_losses[-1]
                final_accuracy = train_accuracies[-1]

                config_dict = {
                    "dataset": config.dataset_name,
                    "optimizer": config.optimizer_config.optimizer_name,
                    "scheduler": config.scheduler_config.scheduler_name,
                    "batch_size": config.batch_size,
                    "gradient_counter_stop": config.gradient_counter_stop,
                }

                analyzer.analyze_model(
                    model=model,
                    config=config_dict,
                    final_loss=final_loss,
                    final_accuracy=final_accuracy,
                    initialization_type="xavier_uniform",
                )

        return train_losses, train_accuracies

    def _get_optimizer(
        self,
        optimizer_name: str,
    ) -> callable:
        match optimizer_name:
            case "adam":
                return torch.optim.Adam
            case "adamw":
                return torch.optim.AdamW
            case "sgd":
                return torch.optim.SGD
            case "rmsprop":
                return torch.optim.RMSprop
            case _:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")

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
