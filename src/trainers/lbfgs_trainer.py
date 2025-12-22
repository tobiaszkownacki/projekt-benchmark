from typing import override
from src.trainers.base_trainer import BaseTrainer
from src.config import BenchmarkConfig
from src.logging import Log
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss


class LbfgsTrainer(BaseTrainer):
    MAX_SAMPLES = 5000  # due to computational/memory limit
    
    @override
    def train(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.TensorDataset,
        config: BenchmarkConfig,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        n_total = len(train_dataset)
        use_samples = min(n_total, self.MAX_SAMPLES)
        
        if n_total > self.MAX_SAMPLES:
            print(f"LBFGS: Subsampling {n_total} to {use_samples} samples for memory efficiency")
        
        if isinstance(train_dataset, TensorDataset):
            # TensorDataset
            inputs_full = train_dataset.tensors[0][:use_samples]
            targets_full = train_dataset.tensors[1][:use_samples]
        else:
            # ConcatDataset
            inputs_list = []
            targets_list = []
            loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
            collected = 0
            for batch_x, batch_y in loader:
                inputs_list.append(batch_x)
                targets_list.append(batch_y)
                collected += len(batch_x)
                if collected >= use_samples:
                    break
            inputs_full = torch.cat(inputs_list, dim=0)[:use_samples]
            targets_full = torch.cat(targets_list, dim=0)[:use_samples]
        
        inputs_full = inputs_full.to(device)
        targets_full = targets_full.to(device)
        
        criterion = CrossEntropyLoss()
        optimizer = self._get_optimizer()(model.parameters(), **self.params)
        log = Log(
            output_file=f"{self.__class__.__name__}-"
            f"{config.dataset_name}-"
            f"{self.name}-"
            f"fullbatch.csv"
        )
        gradient_counter = 0
        train_losses = []
        train_accuracies = []
        
        model.to(device)
        
        # Freeze BatchNorm layers - conflict with LBFGS
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
        
        step_counter = 0
        stop_training = False
        while not stop_training:            
            def closure():
                nonlocal gradient_counter, stop_training
                if gradient_counter >= config.gradient_counter_stop:
                    stop_training = True
                    with torch.no_grad():
                        output = model(inputs_full)
                        return criterion(output, targets_full)
                
                optimizer.zero_grad()
                output = model(inputs_full)
                loss = criterion(output, targets_full)
                loss.backward()
                gradient_counter += 1
                return loss
            
            loss = optimizer.step(closure)
            step_counter += 1
            
            if stop_training:
                break
            
            with torch.no_grad():
                train_output = model(inputs_full)
                train_loss = criterion(train_output, targets_full).item()
                _, train_predicted = torch.max(train_output, 1)
                train_acc = 100 * (train_predicted == targets_full).sum().item() / len(targets_full)
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            log.add_number_of_samples(len(targets_full))
            log.log(round(train_loss, 4), config.save_interval)
            
            print(f"Step {step_counter}: Gradients={gradient_counter}, Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%")
        
        return train_losses, train_accuracies

    @override
    def _get_optimizer(self) -> callable:
        match self.name:
            case "lbfgs":
                return torch.optim.LBFGS
            case _:
                raise ValueError(f"Unsupported optimizer: {self.name}")

    @override
    def _get_optimizer_params(self, optimizer_params) -> dict:
        params = {
            "lr": optimizer_params.lr,
            "history_size": 10,  # reduced from default for memory efficiency
            "max_iter": 10,
            "line_search_fn": "strong_wolfe",
        }
        return params