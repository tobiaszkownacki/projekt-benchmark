from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Type, Callable
from enum import Enum, auto


@dataclass
class BenchmarkResult:
    optimizer_name: str
    dataset_name: str
    
    # Stop info
    total_steps: int
    total_epochs: int
    
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
            "steps": self.total_steps,
            "epochs": self.total_epochs,
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
