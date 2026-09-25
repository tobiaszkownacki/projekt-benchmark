import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pipeline.executor import ExecutorAdapter
from pipeline.task_repository import TaskRepository

logger = logging.getLogger(__name__)


@dataclass
class CompletionSignal:
    new_completed_tasks_with_success: list[str]
    new_completed_tasks_with_failure: list[str]


class CompletionSource(ABC):
    @abstractmethod
    def on_poll(self) -> CompletionSignal: ...


@dataclass(frozen=True)
class JobState:
    executor_task_id: str
    job_name: str
    succeeded: bool
    failed: bool
    running: bool = False


class PollableExecutor(ExecutorAdapter):
    """An executor whose jobs have to be watched to find out they are done.

    Separate from ExecutorAdapter because an executor that reports its own
    completion implements none of it. The state vocabulary stays on this side:
    SLURM's TIMEOUT means nothing to another backend, so an executor collapses
    its own states into succeeded / failed / running.
    """

    @abstractmethod
    def poll_job_states(self) -> list[JobState]: ...
    @abstractmethod
    def fetch_error_tail(self, job_name: str, executor_task_id: str) -> str: ...


class PolledCompletionSource(CompletionSource):
    """Reconciles observed job states against the task table.

    Every write in the poll path is here rather than in an adapter, so a new
    executor contributes a way to watch jobs and inherits the state machine
    instead of a second copy of it.
    """

    def __init__(self, adapter: PollableExecutor, task_repo: TaskRepository) -> None:
        self.adapter = adapter
        self.task_repo = task_repo

    def on_poll(self) -> CompletionSignal:
        completed: list[str] = []
        failed: list[str] = []
        for job in self.adapter.poll_job_states():
            if job.succeeded:
                task_id = self._reconcile_completed(job)
                if task_id:
                    completed.append(task_id)
            elif job.failed:
                task_id = self._reconcile_failed(job)
                if task_id:
                    failed.append(task_id)
            elif job.running:
                self.task_repo.mark_running_by_executor_id(job.executor_task_id)
        return CompletionSignal(completed, failed)

    def _reconcile_completed(self, job: JobState) -> str | None:
        record = self.task_repo.get_by_executor_id(job.executor_task_id)
        if record is None:
            # The only symptom an orphaned job on the cluster ever produces.
            logger.warning(f"executor_task_id={job.executor_task_id} finished but has no matching task")
            return None
        if record.task_status == "COMPLETED":
            return None
        if not self.task_repo.mark_completed_by_executor_id(job.executor_task_id):
            return None
        logger.info(f"task_id={record.task_id} completed (executor_task_id={job.executor_task_id})")
        return record.task_id

    def _reconcile_failed(self, job: JobState) -> str | None:
        record = self.task_repo.get_by_executor_id(job.executor_task_id)
        if record is None:
            logger.warning(f"executor_task_id={job.executor_task_id} failed but has no matching task")
            return None
        if record.task_status == "FAILED":
            return None
        self.task_repo.mark_failed(record.task_id, self.adapter.fetch_error_tail(job.job_name, job.executor_task_id))
        logger.error(f"task_id={record.task_id} failed (executor_task_id={job.executor_task_id})")
        return record.task_id
