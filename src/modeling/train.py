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
from src.modeling.predict import cma_predict


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
    starting_accuracy = cma_predict(model, train_loader)
    print(f"Starting accuracy: {starting_accuracy * 100:.2f}%")
    losses_per_epoch, accuracies_per_epoch = train_cma_batch_based(
        model=model,
        train_loader=train_loader,
        config=config,
        es=es,
        criterion=criterion,
    )

    return losses_per_epoch, accuracies_per_epoch


def train_cma_batch_based(
    model: torch.nn.Module,
    train_loader: TensorDataset,
    config: Config,
    es: cma.CMAEvolutionStrategy,
    criterion: nn.Module,
):
    epoch_count = 0
    losses_per_epoch = []
    accuracies_per_epoch = []
    while epoch_count <= config.max_epochs and not es.stop():
        model.eval()
        loss_in_epoch = 0.0
        with torch.no_grad():
            for inputs, targets in train_loader:
                losses = []
                correct = 0
                solutions = es.ask()
                for s in solutions:
                    cma_set_params(model, np.array(s))

                    outputs = model(inputs)
                    loss = criterion(outputs, targets).item() / inputs.size(0)

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    batch_accuracy = (predicted == targets).sum().item() / targets.size(0)
                    losses.append(loss)
                es.tell(solutions, losses)
                print(f'Average loss per one batch {sum(losses):.2f}')
                loss_in_epoch += sum(losses) / len(losses)
                print(f"Batch accuracy: {batch_accuracy * 100:.2f}%")
            epoch_count += 1
            losses_per_epoch.append(loss_in_epoch / len(train_loader))
            cma_set_params(model, es.best.x)
            best_accuracy = cma_predict(model, train_loader)
            print("\nEPOCH SUMMARY")
            print(f"Loss in the last epoch: {loss_in_epoch / len(train_loader):.2f}")
            print(f"CMA-ES model accuracy: {best_accuracy * 100:.2f}%")
            print(f"The lowest loss: {es.best.f:.2f}\n")
            accuracies_per_epoch.append(best_accuracy)

    return losses_per_epoch, accuracies_per_epoch


def cma_set_params(model: torch.nn.Module, best_params: np.ndarray):
    idx = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(
            torch.from_numpy(best_params[idx: idx + numel].reshape(param.shape))
        )
        idx += numel


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
        max_epochs=args.max_epochs,
    )
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    model = DATA_SETS[config.dataset_name]["model"]
    data_set = DATA_SETS[config.dataset_name]["data_set"]()
    optimizer = OptimizerFactory.get_optimizer(config.optimizer_config.optimizer_name)
    training_function = select_training(config)
    training_function(model=model(), train_dataset=data_set, config=config, optimizer=optimizer)


if __name__ == "__main__":
    main(sys.argv)
