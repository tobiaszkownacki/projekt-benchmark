from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Type, Callable
from enum import Enum, auto

class StopReason(Enum):
    GRADIENT_LIMIT = auto()
    DATABASE_LIMIT = auto()
    EPOCH_LIMIT = auto()
    OPTIMIZER_CONVERGED = auto()
    MAX_STEPS = auto()


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    optimizer_name: str
    dataset_name: str
    
    # Stop info
    stop_reason: StopReason
    total_steps: int
    total_epochs: int
    wall_time_seconds: float
    
    # Metrics
    final_loss: float
    final_accuracy: float
    gradient_count: int
    database_reaches: int
    
    # History
    loss_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimizer": self.optimizer_name,
            "dataset": self.dataset_name,
            "stop_reason": self.stop_reason.name,
            "steps": self.total_steps,
            "epochs": self.total_epochs,
            "wall_time": self.wall_time_seconds,
            "final_loss": self.final_loss,
            "final_accuracy": self.final_accuracy,
            "gradient_count": self.gradient_count,
            "database_reaches": self.database_reaches,
        }


@dataclass
class StopCondition:
    """When to stop the benchmark."""
    max_gradient_count: Optional[int] = None
    max_database_reaches: Optional[int] = None
    max_epochs: Optional[int] = None
    max_steps: Optional[int] = None  # steps = optimizer.step() calls
    
    def __post_init__(self):
        if all(v is None for v in [
            self.max_gradient_count,
            self.max_database_reaches, 
            self.max_epochs,
            self.max_steps
        ]):
            raise ValueError("At least one stop condition required")
