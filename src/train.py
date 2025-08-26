import torch
import numpy as np
import os
import glob
from src.config import Config, SchedulerConfig, CMAOptimizerConfig, GradientOptimizerConfig, LBFGSOptimizerConfig, BaseOptimizerConfig
import sys
from datetime import datetime
from src.plots import ModelAnalyzer
from src.arg_parse import get_args
from src.dataset import DATA_SETS
from src.trainers.base_trainer import BaseTrainer
from src.trainers.gradient_trainer import GradientTrainer
from src.trainers.cmaes_trainer import CmaesTrainer
from src.trainers.lbfgs_trainer import LbfgsTrainer


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
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    model = DATA_SETS[config.dataset_name]["model"]()
    load_model = args.load_model
    if load_model:
        model = load_weights(model, load_model)
    save_model = args.save_model
    data_set = DATA_SETS[config.dataset_name]["data_set"]()
    trainer = select_training(config)
    trainer.train(
        model=model,
        train_dataset=data_set,
        config=config,
    )
    if save_model:
        save_dir = 'weights'
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        name = args.optimizer + "_" + args.dataset + "_" + current_time
        torch.save(model.state_dict(), os.path.join(save_dir, name))
        print(f"Model weights saved to: {save_dir}/{name}")


if __name__ == "__main__":
    analyzer = ModelAnalyzer()
    analysis_files = glob.glob("reports/model_analysis/analysis_*.json")
    analyzer.compare_initializations(analysis_files)
    #main(sys.argv)
