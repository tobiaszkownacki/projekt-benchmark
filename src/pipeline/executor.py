from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SubmitResult:
    executor_task_id: str
    std_out: str

@dataclass
class JobDescription:
    task_id: str
    dataset: str #TODO use literal
    optimizers: list[str] #TODO list of literals
    run_name: str

    @classmethod
    def from_message(cls, msg: dict) -> "JobDescription":
        return cls(
            task_id=msg["task_id"],
            dataset=msg["dataset"],
            optimizers=[o.strip() for o in msg["optimizer"].split(",") if o.strip()],
            run_name=msg["run_name"],
        )

@dataclass
class FetchResult:
    files: list[str]

class ExecutorAdapter(ABC):

    @abstractmethod
    def check_ready(self) -> bool:
        ...
    @abstractmethod
    def submit_job(self,job: JobDescription) -> SubmitResult:
        ...
    @abstractmethod
    def fetch_results(self, task_id: str,delete_after_download: bool = False) -> FetchResult:
        ...

