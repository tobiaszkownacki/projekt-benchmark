import torch
import numpy as np
import os
import glob
from src.config import Config, SchedulerConfig, CMAOptimizerConfig, GradientOptimizerConfig, LBFGSOptimizerConfig, BaseOptimizerConfig
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.analyzers.model_analyzer import ModelAnalyzer
from src.arg_parse import get_args
from src.dataset import DATA_SETS
from src.trainers.base_trainer import BaseTrainer
from src.trainers.gradient_trainer import GradientTrainer
from src.trainers.cmaes_trainer import CmaesTrainer
from src.trainers.lbfgs_trainer import LbfgsTrainer
from src.analyzers.model_analyzer import ModelAnalyzer
from torch.utils.data import DataLoader

def select_training(config: Config) -> BaseTrainer:
    match config.optimizer_config:
        case GradientOptimizerConfig():
            return GradientTrainer()
        case CMAOptimizerConfig():
            return CmaesTrainer()
        case LBFGSOptimizerConfig():
            return LbfgsTrainer()
        case _:
            raise ValueError(
                f"Unsupported optimizer configuration: {config.optimizer_config.optimizer_name}"
            )


def get_optimizer_config(optimizer_name: str) -> BaseOptimizerConfig:
    match optimizer_name:
        case name if name in ["adam", "adamw", "sgd", "rmsprop"]:
            return GradientOptimizerConfig(optimizer_name=optimizer_name)
        case "cma-es":
            return CMAOptimizerConfig(optimizer_name=optimizer_name)
        case "lbfgs":
            return LBFGSOptimizerConfig(optimizer_name=optimizer_name)
        case _:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def load_weights(model, path):
    try:
        load_dir = 'weights'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(os.path.join(load_dir, path), map_location=device)

        model_keys = set(model.state_dict().keys())
        state_keys = set(state_dict.keys())
        if not state_keys.issubset(model_keys):
            raise RuntimeError("Weights are not matching the models architecture.")

        model.load_state_dict(state_dict)
        model.eval()
        return model

    except RuntimeError as e:
        print(f"Error while loadign from path: {path}")
        print("Error details:")
        raise e


def evaluate(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:

            device = next(model.parameters()).device
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    
    return avg_loss

def main(arguments):
    args = get_args(arguments)

    scheduler_config = SchedulerConfig(
        scheduler_name=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        patience=args.patience
    )

    config = Config(
        dataset_name=args.dataset,
        scheduler_config=scheduler_config,
        batch_size=args.batch_size,
        reaching_count=args.reaching_count,
        gradient_counter_stop=args.gradient_counter_stop,
        random_seed=args.random_seed,
        optimizer_config=get_optimizer_config(args.optimizer),
        max_epochs=args.max_epochs,
        save_interval=args.save_interval,
        initialization_xavier=args.init_xavier
    )
    np.random.seed(config.random_seed)
    
    load_model = args.load_model
    save_model = args.save_model
    if load_model:
        model = load_weights(model, load_model)
        loss, accuracy = evaluate(model, val_dataset, torch.nn.CrossEntropyLoss())
        print(f"\nFinal Results:")
        print(f"Val Loss: {loss:.4f}")
        print(f"Val Accuracy: {accuracy:.2f}%")
    else:
        analyzer = ModelAnalyzer()

        all_val_losses = []
        val_loss_from_results = []
        accuracy_from_results= []
        number_of_seeds = 50
        for seed in range(1, number_of_seeds+1):
            torch.manual_seed(seed)
            model = DATA_SETS[config.dataset_name]["model"]()
            train_dataset, val_dataset = DATA_SETS[config.dataset_name]["data_set"]()

            trainer = select_training(config)
            results = trainer.train(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=config,
                lr=0.0008
            )
            val_loss_from_results.append(results["val_losses"][-1])
            accuracy_from_results.append(results["val_accuracies"][-1])
            all_val_losses.append(results["val_losses"])
            if save_model:
                save_dir = 'weights'
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
                name = args.optimizer + "_" + args.dataset + "_" + current_time
                torch.save(model.state_dict(), os.path.join(save_dir, name))
                print(f"Model weights saved to: {save_dir}/{name}")

        analyzer.create_n_seed_losses_plot(all_val_losses, config)

        val_loss_mean = np.mean(val_loss_from_results)
        val_loss_max = np.max(val_loss_from_results)
        val_loss_min = np.min(val_loss_from_results)
        val_loss_std = np.std(val_loss_from_results)
        acc_mean = np.mean(accuracy_from_results)
        acc_max = np.max(accuracy_from_results)
        acc_min = np.min(accuracy_from_results)
        acc_std = np.std(accuracy_from_results)

        print(f"\n=== Summary for {number_of_seeds} seeds ===")
        print(f"Val Loss: {val_loss_mean:.4f} ± {val_loss_std:.4f}")
        print(f"Val Loss Max: {val_loss_max:.4f}, Val Loss Min {val_loss_min:.4f}")
        print(f"Val Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}%")
        print(f"Val Accuracy Max: {acc_max:.4f}, Val Accuracy Min {acc_min:.4f}")
        
        analyzer.create_box_plots(val_loss_from_results, accuracy_from_results, config)
        analyzer.create_violin_plots(val_loss_from_results, accuracy_from_results, config)

        analysis_files = glob.glob("reports/model_analysis/analysis_*.json")
        analyzer.compare_initializations(analysis_files)




if __name__ == "__main__":
    main(sys.argv)
