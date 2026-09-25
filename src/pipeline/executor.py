from abc import ABC, abstractmethod
from dataclasses import dataclass

SCHEMA_VERSION = 1


@dataclass
class SubmitResult:
    executor_task_id: str
    std_out: str


@dataclass
class JobDescription:
    """Everything an executor needs to build a run, and nothing it has to look up.

    A compute node has no route to the database, so a run that is not fully
    described by its message is a run only the local half of the system can
    start.
    """

    task_id: str
    dataset: str
    model: str
    optimizers: list[str]
    seed: int
    run_name: str
    stop_condition: dict[str, int]
    webhook_token: str | None = None

    @classmethod
    def from_message(cls, msg: dict) -> "JobDescription":
        version = msg.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"message schema_version={version!r}, this executor reads {SCHEMA_VERSION}")

        optimizers = [name.strip() for name in msg["optimizers"] if name.strip()]
        if len(optimizers) != 1:
            # results is keyed by task_id, so a run with two optimizers has
            # nowhere to put the second number.
            raise ValueError(f"task_id={msg['task_id']} carries {len(optimizers)} optimizers, one run stores one")

        stop_condition = {key: int(value) for key, value in dict(msg["stop_condition"]).items()}
        if not stop_condition:
            raise ValueError(f"task_id={msg['task_id']} carries no stop condition, so the run has no budget")

        return cls(
            task_id=msg["task_id"],
            dataset=msg["dataset"],
            model=msg["model"],
            optimizers=optimizers,
            seed=int(msg["seed"]),
            run_name=msg["run_name"],
            stop_condition=stop_condition,
            webhook_token=msg.get("webhook_token"),
        )


@dataclass
class FetchResult:
    files: list[str]


class ExecutorAdapter(ABC):
    @abstractmethod
    def submit_job(self, job: JobDescription) -> SubmitResult: ...
    @abstractmethod
    def fetch_results(self, task_id: str, delete_after_download: bool = False) -> FetchResult: ...
