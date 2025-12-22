"""
Store useful variables and configuration
"""

from dataclasses import dataclass, field
from pathlib import Path
from omegaconf import MISSING
import random
from typing import Optional
from hydra.core.config_store import ConfigStore
from torch import nn
from enum import Enum
from models.autoencoders.conv_autoencoder import ConvAutoEncoder
from models.autoencoders.dense_autoencoder_image import DenseAutoEncoderImage
from models.autoencoders.dense_autoencoder_tabular import DenseAutoEncoderTabular
from models.autoencoders.variational_autoencoder_image import VariationalAutoEncoderImage
from models.autoencoders.variational_autoencoder_tabular import VariationalAutoEncoderTabular


# from dotenv import load_dotenv

# Load environment variables from .env file if it exists
# load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


class ProblemType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class DatasetType(Enum):
    TABULAR = "tabular"
    IMAGE = "image"


class ModelType(Enum):
    SUPERVISED = "supervised"  # e.g., classic classification or regression
    DENSE_AE = "dense_ae"
    CONV_AE = "conv_ae"
    VARIATIONAL_AE = "variational_ae"


GENERIC_MODELS = {
    (ModelType.DENSE_AE, DatasetType.TABULAR): DenseAutoEncoderTabular,
    (ModelType.DENSE_AE, DatasetType.IMAGE): DenseAutoEncoderImage,
    (ModelType.CONV_AE, DatasetType.IMAGE): ConvAutoEncoder,
    (ModelType.VARIATIONAL_AE, DatasetType.IMAGE): VariationalAutoEncoderImage,
    (ModelType.VARIATIONAL_AE, DatasetType.TABULAR): VariationalAutoEncoderTabular,
}


ALLOWED_DATASETS = ["cifar10", "heart_disease", "wine_quality", "digits", "abalone"]
ALLOWED_OPTIMIZERS = ["adam", "adamw", "sgd", "rmsprop", "lbfgs", "cma-es", "lion"]
ALLOWED_SCHEDULERS = [
    "none",
    "steplr",
    "exponentiallr",
    "reduceonplateau",
    "cosineannealinglr",
]
ALLOWED_CRITERIONS = {
    ProblemType.CLASSIFICATION: ["cross_entropy"],
    ProblemType.REGRESSION: [
        "mse_loss",
        "mae_loss",
    ],
}


@dataclass
class SchedulerConfig:
    name: str = "none"
    step_size: int = 10
    gamma: float = 0.5
    patience: int = 1


@dataclass
class OptimizerParams:
    lr: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-08
    dampening: float = 0
    maximize: bool = False
    alpha: float = 0.99
    centered: bool = False
    # cma-es specific
    sigma: float = 0.5
    population_size: Optional[int] = None
    cma_diagonal: bool = False


@dataclass
class BenchmarkConfig:
    dataset_name: str
    optimizer_trainer: object
    criterion: nn.Module
    model_type: ModelType
    scheduler_config: SchedulerConfig
    batch_size: int
    reaching_count: int
    gradient_counter_stop: int
    random_seed: int
    max_epochs: int
    save_interval: int
    initialization_xavier: bool


@dataclass
class UserConfig:
    dataset: str = MISSING
    optimizer: str = MISSING
    criterion: str = MISSING
    model: ModelType = MISSING

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    optimizer_params: OptimizerParams = field(default_factory=OptimizerParams)
    batch_size: int = 16
    reaching_count: int = 500
    max_epochs: int = 10
    gradient_counter_stop: int = 5000
    random_seed: int = random.randrange(2**32)
    save_interval: int = 10
    save_model: bool = False
    load_model: Optional[str] = None
    init_xavier: bool = False


cs = ConfigStore.instance()
cs.store(name="user_config", node=UserConfig)
