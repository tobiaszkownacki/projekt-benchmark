"""The poller and the completion callback both close tasks, and only the first one counts.

mark_completed_by_executor_id refused only a task that was already COMPLETED
and mark_failed refused nothing, so a poller holding a stale read turned the
callback's FAILED into COMPLETED, or the other way round, and reported the task
as newly closed.
"""

import uuid

import psycopg
import pytest

from pipeline.completion import JobState, PollableExecutor, PolledCompletionSource
from pipeline.executor import FetchResult, JobDescription, SubmitResult
from pipeline.sql_task_repository import SqlTaskRepository


class _Cluster(PollableExecutor):
    def __init__(self, states: list[JobState]) -> None:
        self.states = states

    def submit_job(self, job: JobDescription) -> SubmitResult:
        raise NotImplementedError

    def fetch_results(self, task_id: str, delete_after_download: bool = False) -> FetchResult:
        raise NotImplementedError

    def cancel_job(self, executor_task_id: str) -> None:
        raise NotImplementedError

    def poll_job_states(self) -> list[JobState]:
        return self.states

    def fetch_error_tail(self, job_name: str, executor_task_id: str) -> str:
        return "error tail from the poller"


class _CallbackBetweenReadAndWrite(SqlTaskRepository):
    """The callback lands after the poller read the task and before it writes."""

    def __init__(self, repository: SqlTaskRepository, database_url: str, callback_status: str) -> None:
        super().__init__(repository._db_cls)
        self.database_url = database_url
        self.callback_status = callback_status

    def get_by_executor_id(self, executor_task_id: str):
        record = super().get_by_executor_id(executor_task_id)
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            conn.execute(
                "UPDATE tasks SET task_status = %s, error_message = 'from the callback' "
                "WHERE executor_task_id = %s AND task_status NOT IN ('COMPLETED', 'FAILED')",
                (self.callback_status, executor_task_id),
            )
        return record


def _status(database_url: str, task_id: str) -> tuple[str, str | None]:
    with psycopg.connect(database_url) as conn:
        return conn.execute(
            "SELECT task_status::text, error_message FROM tasks WHERE task_id = %s", (task_id,)
        ).fetchone()


def _job(executor_task_id: str, *, succeeded: bool) -> JobState:
    return JobState(executor_task_id, "job", succeeded=succeeded, failed=not succeeded)


@pytest.mark.parametrize(
    ("callback_status", "poller_succeeded"),
    [("FAILED", True), ("COMPLETED", False)],
    ids=["callback-failed-poller-completed", "callback-completed-poller-failed"],
)
def test_the_poller_does_not_overwrite_what_the_callback_closed_first(
    sql_repository, new_task, database_url, callback_status, poller_succeeded
):
    executor_task_id = f"job-{uuid.uuid4().hex}"
    task_id = new_task("RUNNING", executor_task_id)
    repository = _CallbackBetweenReadAndWrite(sql_repository, database_url, callback_status)
    source = PolledCompletionSource(_Cluster([_job(executor_task_id, succeeded=poller_succeeded)]), repository)

    signal = source.on_poll()

    assert signal.new_completed_tasks_with_success == []
    assert signal.new_completed_tasks_with_failure == []
    assert _status(database_url, task_id) == (callback_status, "from the callback")


def test_a_failed_task_is_not_completed(sql_repository, new_task, database_url):
    executor_task_id = f"job-{uuid.uuid4().hex}"
    task_id = new_task("FAILED", executor_task_id)

    assert not sql_repository.mark_completed_by_executor_id(executor_task_id)
    assert _status(database_url, task_id)[0] == "FAILED"


def test_a_completed_task_is_not_failed(sql_repository, new_task, database_url):
    task_id = new_task("COMPLETED", f"job-{uuid.uuid4().hex}")

    assert not sql_repository.mark_failed(task_id, "late")
    assert _status(database_url, task_id) == ("COMPLETED", None)


@pytest.mark.parametrize("task_status", ["PENDING", "SUBMITTED", "RUNNING"])
def test_an_open_task_can_still_be_failed(sql_repository, new_task, database_url, task_status):
    # PENDING is the worker's case: a submit_job that raised, or a message it could not read.
    task_id = new_task(task_status, None if task_status == "PENDING" else f"job-{uuid.uuid4().hex}")

    assert sql_repository.mark_failed(task_id, "sbatch failed")
    assert _status(database_url, task_id) == ("FAILED", "sbatch failed")


def test_a_poller_that_closes_the_task_itself_still_reports_it(sql_repository, new_task, database_url):
    executor_task_id = f"job-{uuid.uuid4().hex}"
    task_id = new_task("RUNNING", executor_task_id)
    source = PolledCompletionSource(_Cluster([_job(executor_task_id, succeeded=False)]), sql_repository)

    signal = source.on_poll()

    assert signal.new_completed_tasks_with_failure == [task_id]
    assert _status(database_url, task_id) == ("FAILED", "error tail from the poller")
