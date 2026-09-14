import logging

from pipeline.completion import CompletionSignal, CompletionSource
from pipeline.executor import ExecutorAdapter, FetchResult, JobDescription, SubmitResult

logger = logging.getLogger(__name__)


class FakeExecutor(ExecutorAdapter):
    def check_ready(self) -> bool:
        return True

    def submit_job(self, job: JobDescription) -> SubmitResult:
        return SubmitResult(executor_task_id="123", std_out="")

    def fetch_results(self, task_id: str, delete_after_download: bool = False) -> FetchResult:
        return FetchResult(files=[])


class FakeCompletionSource(CompletionSource):
    def __init__(self, adapter: FakeExecutor, task_repo) -> None:
        self.adapter = adapter
        self.task_repo = task_repo

    def on_poll(self) -> CompletionSignal:
        return CompletionSignal([], [])
