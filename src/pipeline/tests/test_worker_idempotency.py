"""Delivery out of the outbox is at-least-once, so the worker sees repeats.

Without a guard the second delivery started a second sbatch and overwrote the
identifier of the first, leaving a job on the cluster that nothing tracks. With
more than one worker the two deliveries also overlap, and both used to read an
empty executor_task_id before either had written one.
"""

import threading
import uuid

from fakes import RecordingExecutor, RecordingRepository

from pipeline.executor import SCHEMA_VERSION, JobDescription, SubmitResult
from pipeline.generic_services.generic_worker import GenericWorker
from pipeline.task_repository import TaskStatus
from shared.queue_topology import QueueTopology


def _message(task_id: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "run_name": "run",
        "dataset": "wine_quality",
        "model": "mlp-2x32",
        "optimizers": ["adam"],
        "seed": 11,
        "stop_condition": {"max_epochs": 5},
    }


def _worker(repo: RecordingRepository, adapter: RecordingExecutor) -> GenericWorker:
    return GenericWorker(adapter, QueueTopology("test"), repo, message_broker=None)


def test_the_same_message_twice_submits_one_job():
    repo, adapter = RecordingRepository(), RecordingExecutor()
    task_id = str(uuid.uuid4())
    repo.tasks[task_id] = TaskStatus(task_id, "PENDING")
    worker = _worker(repo, adapter)

    worker.handle(_message(task_id))
    worker.handle(_message(task_id))

    assert len(adapter.submitted) == 1
    assert repo.submitted == [(task_id, "job-1")]


def test_a_task_the_callback_closed_first_is_not_resubmitted():
    repo, adapter = RecordingRepository(), RecordingExecutor()
    task_id = str(uuid.uuid4())
    repo.tasks[task_id] = TaskStatus(task_id, "COMPLETED", "job-1")

    _worker(repo, adapter).handle(_message(task_id))

    assert adapter.submitted == []


class _HeldExecutor(RecordingExecutor):
    """Stays inside submit_job until released, the way an sbatch over SSH does."""

    def __init__(self) -> None:
        super().__init__()
        self.inside = threading.Event()
        self.release = threading.Event()

    def submit_job(self, job: JobDescription) -> SubmitResult:
        self.inside.set()
        assert self.release.wait(5)
        return super().submit_job(job)


def test_two_workers_given_the_same_message_at_once_submit_one_job():
    repo, adapter = RecordingRepository(), _HeldExecutor()
    task_id = str(uuid.uuid4())
    repo.tasks[task_id] = TaskStatus(task_id, "PENDING")
    first = threading.Thread(target=_worker(repo, adapter).handle, args=(_message(task_id),))
    second = threading.Thread(target=_worker(repo, adapter).handle, args=(_message(task_id),))

    first.start()
    assert adapter.inside.wait(5)
    second.start()
    second.join(0.2)
    adapter.release.set()
    first.join(5)
    second.join(5)

    assert len(adapter.submitted) == 1
    assert repo.submitted == [(task_id, "job-1")]
    assert adapter.cancelled == []


class _OutracedRepository(RecordingRepository):
    """Another writer records its own job id between submit_job and mark_submitted."""

    def __init__(self, recorded_by_other: str) -> None:
        super().__init__()
        self.recorded_by_other = recorded_by_other

    def mark_submitted(self, task_id: str, executor_task_id: str) -> bool:
        self.tasks[task_id] = TaskStatus(task_id, "SUBMITTED", self.recorded_by_other)
        return False


def test_a_job_whose_id_lost_the_write_is_cancelled():
    repo, adapter = _OutracedRepository(recorded_by_other="job-0"), RecordingExecutor()
    task_id = str(uuid.uuid4())
    repo.tasks[task_id] = TaskStatus(task_id, "PENDING")

    _worker(repo, adapter).handle(_message(task_id))

    assert adapter.cancelled == ["job-1"]


def test_a_job_the_callback_recorded_first_is_kept():
    repo, adapter = _OutracedRepository(recorded_by_other="job-1"), RecordingExecutor()
    task_id = str(uuid.uuid4())
    repo.tasks[task_id] = TaskStatus(task_id, "PENDING")

    _worker(repo, adapter).handle(_message(task_id))

    assert adapter.cancelled == []
