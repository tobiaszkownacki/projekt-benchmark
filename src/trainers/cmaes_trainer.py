from src.trainers.base_trainer import BaseTrainer
from src.config import Config
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import cma
import numpy as np


class CmaesTrainer(BaseTrainer):
    def train(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.TensorDataset,
        config: Config,
    ):
        all_params = np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in model.parameters()]
        )
        sigma = 0.5
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        criterion = CrossEntropyLoss()
        opts = {'seed': config.random_seed}
        optimizer = self._get_optimizer(config.optimizer_config.optimizer_name)(
            all_params, sigma, opts
        )
        starting_accuracy = self.predict(model, train_loader)
        print(f"Starting accuracy: {starting_accuracy * 100:.2f}%")

        epoch_count = 0
        losses_per_epoch = []
        accuracies_per_epoch = []
        while epoch_count <= config.max_epochs and not optimizer.stop():
            model.eval()
            loss_in_epoch = 0.0
            with torch.no_grad():
                for inputs, targets in train_loader:
                    losses = []
                    correct = 0
                    solutions = optimizer.ask()
                    for s in solutions:
                        self._set_params(model, np.array(s))

                        outputs = model(inputs)
                        loss = criterion(outputs, targets).item() / inputs.size(0)

                        _, predicted = torch.max(outputs, 1)
                        correct += (predicted == targets).sum().item()
                        batch_accuracy = (predicted == targets).sum().item() / targets.size(0)
                        losses.append(loss)
                    optimizer.tell(solutions, losses)
                    print(f'Average loss per one batch {sum(losses):.2f}')
                    loss_in_epoch += sum(losses) / len(losses)
                    print(f"Batch accuracy: {batch_accuracy * 100:.2f}%")
                epoch_count += 1
                losses_per_epoch.append(loss_in_epoch / len(train_loader))
                self._set_params(model, optimizer.best.x)
                best_accuracy = self.predict(model, train_loader)
                print("\nEPOCH SUMMARY")
                print(f"Loss in the last epoch: {loss_in_epoch / len(train_loader):.2f}")
                print(f"CMA-ES model accuracy: {best_accuracy * 100:.2f}%")
                print(f"The lowest loss: {optimizer.best.f:.2f}\n")
                accuracies_per_epoch.append(best_accuracy)

        return losses_per_epoch, accuracies_per_epoch

    def _get_optimizer(
        self,
        optimizer_name: str,
    ) -> callable:
        match optimizer_name:
            case "cma-es":
                return cma.CMAEvolutionStrategy
            case _:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def predict(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
    ) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in train_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        best_accuracy = correct / total
        return best_accuracy

    def _set_params(
        self,
        model: torch.nn.Module,
        params: np.ndarray,
    ):
        idx = 0
        for param in model.parameters():
            numel = param.numel()
            param.data.copy_(
                torch.from_numpy(params[idx: idx + numel].reshape(param.shape))
            )
            idx += numel
