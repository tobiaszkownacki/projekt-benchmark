import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import cma
import numpy as np
from src.config import Config, CMAOptimizerConfig, GradientOptimizerConfig, LBFGSOptimizerConfig
import sys
from src.arg_parse import get_args
from src.dataset import DATA_SETS
from tqdm import tqdm
from src.optimizers import OptimizerFactory


def select_training(config: Config) -> callable:
    if isinstance(config.optimizer_config, GradientOptimizerConfig):
        return train_gradient
    if isinstance(config.optimizer_config, CMAOptimizerConfig):
        return train_cma
    if isinstance(config.optimizer_config, LBFGSOptimizerConfig):
        return train_lbfgs

    raise ValueError(
        f"Unsupported optimizer configuration: {config.optimizer_config.optimizer_name}"
    )


def train_gradient(
    model: torch.nn.Module,
    train_dataset: TensorDataset,
    config: Config,
    optimizer: torch.optim.Optimizer
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer_instance = optimizer(model.parameters())
    gradient_counter = 0
    train_losses = []
    train_accuracies = []

    while gradient_counter <= config.gradient_counter_stop:
        model.train()
        model.to(device)
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            if gradient_counter > config.gradient_counter_stop:
                break

            optimizer_instance.zero_grad()

            output = model(inputs)
            loss = criterion(output, targets)
            gradient_counter += 1
            loss = loss.to(device)
            loss.backward()
            optimizer_instance.step()

            running_loss += loss.item()

            _, predicted = torch.max(output, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        print(f"Gradient counter: {gradient_counter}")
        train_loss = running_loss / total
        train_losses.append(train_loss)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        print(f"Train loss: {train_loss}, Train accuracy: {train_accuracy}")
    return train_losses, train_accuracies


def train_lbfgs(
    model: torch.nn.Module,
    train_dataset: TensorDataset,
    config: Config,
    optimizer: torch.optim.Optimizer
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer_instance = optimizer(model.parameters())
    gradient_counter = 0
    train_losses = []
    train_accuracies = []

    while gradient_counter <= config.gradient_counter_stop:
        model.train()
        model.to(device)
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            if gradient_counter > config.gradient_counter_stop:
                break

            def closure():
                optimizer_instance.zero_grad()
                output = model(inputs)
                loss = criterion(output, targets)
                loss.backward()
                return loss

            loss = optimizer_instance.step(closure)
            gradient_counter += 1

            with torch.no_grad():
                output = model(inputs)
                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print(f"Gradient counter: {gradient_counter}")
        train_loss = running_loss / total
        train_losses.append(train_loss)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        print(f"Train loss: {train_loss}, Train accuracy: {train_accuracy}")
    return train_losses, train_accuracies


def train_cma(
    model: torch.nn.Module,
    train_dataset: TensorDataset,
    config: Config,
    optimizer: cma.CMAEvolutionStrategy,
):
    all_params = np.concatenate(
        [p.detach().cpu().numpy().ravel() for p in model.parameters()]
    )
    print(len(all_params))
    sigma = 0.5
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    opts = {
        'seed': config.random_seed,
    }
    es = optimizer(all_params, sigma, opts)

    reaching_count = 0
    losses_per_reach = []
    accuracies_per_reach = []

    while reaching_count <= config.reaching_count and not es.stop():
        solutions = es.ask()
        losses = []

        for s in solutions:
            idx = 0
            flat_params = np.array(s)
            for param in model.parameters():
                numel = param.numel()
                param.data.copy_(
                    torch.from_numpy(
                        flat_params[idx: idx + numel].reshape(param.shape)
                    )
                )
                idx += numel

            model.eval()
            total_loss = 0.0
            total = 0
            correct = 0
            with torch.no_grad():
                for inputs, targets in train_loader:
                    # if reaching_count >= config.reaching_count:
                    #     break
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    batch_loss = loss.item()
                    total_loss += loss.item() * inputs.size(0)

                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                    batch_accuracy = (predicted == targets).sum().item() / targets.size(
                        0
                    )
                    accuracies_per_reach.append(batch_accuracy)
                    losses_per_reach.append(batch_loss)
                    reaching_count += 1
            if total != 0:
                avg_loss = total_loss / total
                losses.append(avg_loss)
            else:
                losses.append(0)
        print(f'Average loss per one epoch {sum(losses) / len(losses)}')
        print(f'Reaching_count: {reaching_count}')
        es.tell(solutions, losses)

    # Ustawienie w modelu najlepszych parametrów
    best_params = es.best.x
    idx = 0
    flat_params = np.array(s)
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(
            torch.from_numpy(best_params[idx: idx + numel].reshape(param.shape))
        )
        idx += numel
    # Ocena accuracy modelu
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
    print(f"Best CMA-ES model accuracy: {best_accuracy * 100:.2f}%")

    return losses_per_reach, accuracies_per_reach


def main(arguments):
    args = get_args(arguments)
    config = Config(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        reaching_count=args.reaching_count,
        gradient_counter_stop=args.gradient_counter_stop,
        random_seed=args.random_seed,
        optimizer_config=(
            CMAOptimizerConfig(args.optimizer)
            if args.optimizer in ["cma-es"]
            else LBFGSOptimizerConfig(args.optimizer)
            if args.optimizer in ["lbfgs"]
            else GradientOptimizerConfig(args.optimizer)
        ),
    )
    if config.random_seed is not None:
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

    model = DATA_SETS[config.dataset_name]["model"]
    data_set = DATA_SETS[config.dataset_name]["data_set"]()
    optimizer = OptimizerFactory.get_optimizer(config.optimizer_config.optimizer_name)
    training_function = select_training(config)
    training_function(model=model(), train_dataset=data_set, config=config, optimizer=optimizer)


if __name__ == "__main__":
    main(sys.argv)
