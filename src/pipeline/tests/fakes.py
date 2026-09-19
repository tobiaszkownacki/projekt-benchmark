"""Stand-ins for the database and the cluster, shared by the pipeline tests."""

from pipeline.executor import ExecutorAdapter, FetchResult, JobDescription, SubmitResult
from pipeline.task_repository import TaskRepository, TaskStatus
from shared.run_result import RunResult


class RecordingRepository(TaskRepository):
    def __init__(self, tasks: dict[str, TaskStatus] | None = None) -> None:
        self.tasks = tasks or {}
        self.submitted: list[tuple[str, str]] = []
        self.failed: list[tuple[str, str]] = []
        self.errors: list[tuple[str, str]] = []
        self.running: list[str] = []
        self.results: dict[str, RunResult] = {}
        self.artifacts: dict[str, tuple[int, int]] = {}

    def mark_submitted(self, task_id: str, executor_task_id: str) -> bool:
        record = self.tasks.get(task_id)
        if record is not None and record.executor_task_id:
            return False
        self.submitted.append((task_id, executor_task_id))
        self.tasks[task_id] = TaskStatus(task_id, "SUBMITTED", executor_task_id)
        return True

    def mark_failed(self, task_id: str, error_message: str) -> None:
        self.failed.append((task_id, error_message))

    def set_error(self, task_id: str, error_message: str) -> None:
        self.errors.append((task_id, error_message))

    def mark_completed_by_executor_id(self, executor_task_id: str) -> bool:
        return False

    def get_by_executor_id(self, executor_task_id: str) -> TaskStatus | None:
        for record in self.tasks.values():
            if record.executor_task_id == executor_task_id:
                return record
        return None

    def get_by_task_id(self, task_id: str) -> TaskStatus | None:
        return self.tasks.get(task_id)

    def mark_running_by_executor_id(self, executor_task_id: str) -> bool:
        self.running.append(executor_task_id)
        return True

    def store_result(self, task_id: str, result: RunResult) -> None:
        self.results[task_id] = result

    def mark_artifacts(self, task_id: str, files: int, total_bytes: int) -> None:
        self.artifacts[task_id] = (files, total_bytes)


class RecordingExecutor(ExecutorAdapter):
    def __init__(self, executor_task_id: str = "job-1") -> None:
        self.executor_task_id = executor_task_id
        self.submitted: list[JobDescription] = []
        self.fetched: list[str] = []

    def submit_job(self, job: JobDescription) -> SubmitResult:
        self.submitted.append(job)
        return SubmitResult(executor_task_id=self.executor_task_id, std_out="")

    def fetch_results(self, task_id: str, delete_after_download: bool = False) -> FetchResult:
        self.fetched.append(task_id)
        return FetchResult(files=[])
