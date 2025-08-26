"""
Store useful variables and configuration
"""
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class BaseOptimizerConfig():
    optimizer_name: str


@dataclass
class GradientOptimizerConfig(BaseOptimizerConfig):
    optimizer_name = "adam"


@dataclass
class CMAOptimizerConfig(BaseOptimizerConfig):
    optimizer_name = "cma-es"


@dataclass
class LBFGSOptimizerConfig(BaseOptimizerConfig):
    optimizer_name = "lbfgs"


@dataclass
class SchedulerConfig:
    scheduler_name: str
    step_size: int = 10
    gamma: float = 0.5
    patience: int = 5


@dataclass
class Config:
    dataset_name: str
    optimizer_config: GradientOptimizerConfig | CMAOptimizerConfig | LBFGSOptimizerConfig
    scheduler_config: SchedulerConfig
    batch_size: int
    reaching_count: int
    gradient_counter_stop: int
    random_seed: int
    max_epochs: int
    save_interval: int
    initialization_xavier: bool
