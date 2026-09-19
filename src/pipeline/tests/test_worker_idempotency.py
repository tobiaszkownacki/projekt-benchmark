"""Delivery out of the outbox is at-least-once, so the worker sees repeats.

Without a guard the second delivery started a second sbatch and overwrote the
identifier of the first, leaving a job on the cluster that nothing tracks.
"""

import uuid

from fakes import RecordingExecutor, RecordingRepository

from pipeline.executor import SCHEMA_VERSION
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
