import torch
import numpy as np
import os
from src.config import Config, CMAOptimizerConfig, GradientOptimizerConfig, LBFGSOptimizerConfig, BaseOptimizerConfig
import sys
from datetime import datetime
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


def main(arguments):
    args = get_args(arguments)
    config = Config(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        reaching_count=args.reaching_count,
        gradient_counter_stop=args.gradient_counter_stop,
        random_seed=args.random_seed,
        optimizer_config=get_optimizer_config(args.optimizer),
        max_epochs=args.max_epochs,
        save_interval=args.save_interval
    )
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    model = DATA_SETS[config.dataset_name]["model"]()
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
    main(sys.argv)
