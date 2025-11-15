from dataclasses import MISSING
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import os
from src.config import (
    BenchmarkConfig,
    CMAOptimizerConfig,
    GradientOptimizerConfig,
    LBFGSOptimizerConfig,
    BaseOptimizerConfig,
)
from datetime import datetime
from src.config import ALLOWED_DATASETS, ALLOWED_OPTIMIZERS, ALLOWED_SCHEDULERS, UserConfig
from src.dataset import DATA_SETS
from src.trainers.base_trainer import BaseTrainer
from src.trainers.gradient_trainer import GradientTrainer
from src.trainers.cmaes_trainer import CmaesTrainer
from src.trainers.lbfgs_trainer import LbfgsTrainer



def select_training(config: BenchmarkConfig) -> BaseTrainer:
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
        load_dir = "weights"
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


def validate_input(cfg: UserConfig):
    if cfg.dataset is MISSING or cfg.optimizer is MISSING:
        raise ValueError("Both dataset and optimizer must be specified.")
    if cfg.dataset not in ALLOWED_DATASETS:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")
    if cfg.optimizer not in ALLOWED_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")
    if cfg.scheduler.name not in ALLOWED_SCHEDULERS:
        raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")

    # print cfg to yaml
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))


@hydra.main(version_base=None, config_name="user_config")
def main(cfg: UserConfig):

    cfg = OmegaConf.merge(UserConfig(), cfg)
    validate_input(cfg)
    config = BenchmarkConfig(
        dataset_name=cfg.dataset,
        scheduler_config=cfg.scheduler,
        batch_size=cfg.batch_size,
        reaching_count=cfg.reaching_count,
        gradient_counter_stop=cfg.gradient_counter_stop,
        random_seed=cfg.random_seed,
        optimizer_config=get_optimizer_config(cfg.optimizer),
        max_epochs=cfg.max_epochs,
        save_interval=cfg.save_interval,
        initialization_xavier=cfg.init_xavier,
    )
    
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    model = DATA_SETS[config.dataset_name]["model"]()
    load_model = cfg.load_model
    if load_model:
        model = load_weights(model, load_model)
    save_model = cfg.save_model
    data_set = DATA_SETS[config.dataset_name]["data_set"]()
    trainer = select_training(config)
    trainer.train(
        model=model,
        train_dataset=data_set,
        config=config,
    )
    if save_model:
        save_dir = "weights"
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        name = cfg.optimizer + "_" + cfg.dataset + "_" + current_time
        torch.save(model.state_dict(), os.path.join(save_dir, name))
        print(f"Model weights saved to: {save_dir}/{name}")


if __name__ == "__main__":
    main()
