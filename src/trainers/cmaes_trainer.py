from typing import override
from src.trainers.base_trainer import BaseTrainer
from src.config import BenchmarkConfig
from src.logging import Log
import torch
from torch.utils.data import DataLoader
import cma
import numpy as np
from torch.nn.utils import parameters_to_vector


class CmaesTrainer(BaseTrainer):
    @override
    def train(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.TensorDataset,
        config: BenchmarkConfig,
    ):
        vec_params = parameters_to_vector(model.parameters())
        all_params = vec_params.detach().cpu().numpy()

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        sigma = self.params["sigma"]
        opts = {
            "seed": config.random_seed,
            **{k: v for k, v in self.params.items() if k != "sigma"},
        }
        optimizer = self._get_optimizer(self.name)(all_params, sigma, opts)
        criterion = config.criterion()

        log = Log(
            output_file=f"{self.__class__.__name__}-"
            f"{config.dataset_name}-"
            f"{self.name}-"
            f"{config.batch_size}.csv"
        )
        epoch_count = 0
        while epoch_count <= config.max_epochs and not optimizer.stop():
            model.eval()
            epoch_loss = 0.0

            with torch.no_grad():
                for inputs, targets in train_loader:
                    losses = []
                    batch_avg_loss = 0.0
                    solutions = optimizer.ask()
                    for s in solutions:
                        self._set_params(model, np.array(s))

                        loss, outputs = self._calculate_step(
                            model, inputs, targets, criterion, config.model_type
                        )
                        losses.append(loss)

                    optimizer.tell(solutions, losses)
                    print(f"Total loss over population for batch {sum(losses):.2f}")
                    batch_avg_loss += sum(losses) / len(losses)
                    epoch_loss += batch_avg_loss

                    log.add_number_of_samples(targets.size(0))
                    log.log(round(batch_avg_loss, 4), config.save_interval)

                    print(f"Batch average loss (population): {batch_avg_loss:.4f}\n")

                epoch_count += 1
                epoch_avg_loss = epoch_loss / len(train_loader)

                print("\nEPOCH SUMMARY")
                print(f"Avg loss in epoch: {epoch_avg_loss:.2f}")
                print(f"The lowest loss: {optimizer.best.f:.2f}\n")

    @override
    def _get_optimizer(
        self,
        optimizer_name: str,
    ) -> callable:
        match optimizer_name:
            case "cma-es":
                return cma.CMAEvolutionStrategy
            case _:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    @override
    def _get_optimizer_params(
        self,
        optimizer_params,
    ) -> dict:
        self.sigma = optimizer_params.sigma
        dict_params = {
            "sigma": optimizer_params.sigma,
            "CMA_diagonal": optimizer_params.cma_diagonal,
        }
        if optimizer_params.population_size is not None:
            dict_params["popsize"] = optimizer_params.population_size
        return dict_params

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
                torch.from_numpy(params[idx : idx + numel].reshape(param.shape))
            )
            idx += numel
