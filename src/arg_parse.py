from dataclasses import dataclass, field
import random
from typing import Optional
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

ALLOWED_DATASETS = ["cifar10", "heart_disease", "wine_quality", "digits"]
ALLOWED_OPTIMIZERS = ["adam", "adamw", "sgd", "rmsprop", "lbfgs", "cma-es"]
ALLOWED_SCHEDULERS = ["none", "steplr", "exponentiallr", "reduceonplateau", "cosineannealinglr"]


@dataclass
class SchedulerConfig:
    name: str = "none"
    step_size: int = 10
    gamma: float = 0.5
    patience: int = 1


@dataclass
class UserConfig:
    dataset: str = MISSING
    optimizer: str = MISSING

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
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
