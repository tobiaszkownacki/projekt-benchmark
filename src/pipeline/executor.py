from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SubmitResult:
    executor_task_id: str
    std_out: str


@dataclass
class JobDescription:
    """Everything an executor needs to build a run, and nothing it has to look up.

    An executor reads no database row: the cluster side of the system cannot
    reach Postgres, so a job that is not fully described by its message is a job
    that only the local half can start.
    """

    task_id: str
    dataset: str  # TODO use literal
    model: str
    optimizer: str
    seed: int
    run_name: str
    stop_condition: dict[str, int]

    @classmethod
    def from_message(cls, msg: dict) -> "JobDescription":
        optimizer = msg["optimizer"].strip()
        if not optimizer:
            raise ValueError(f"task_id={msg['task_id']} carries no optimizer name")
        return cls(
            task_id=msg["task_id"],
            dataset=msg["dataset"],
            model=msg["model"],
            optimizer=optimizer,
            seed=int(msg["seed"]),
            run_name=msg["run_name"],
            stop_condition={key: int(value) for key, value in dict(msg["stop_condition"]).items()},
        )


@dataclass
class FetchResult:
    files: list[str]


class ExecutorAdapter(ABC):
    @abstractmethod
    def check_ready(self) -> bool: ...
    @abstractmethod
    def submit_job(self, job: JobDescription) -> SubmitResult: ...
    @abstractmethod
    def fetch_results(self, task_id: str, delete_after_download: bool = False) -> FetchResult: ...
