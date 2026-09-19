"""A message the worker cannot read must not leave its task waiting for ever.

The decoder ran outside the try block, so a message missing a field was
dead-lettered without a single write: the submission stayed PENDING, nothing
consumes the dead-letter queue, and the participant saw a run queued since last
week.
"""

import uuid

import pytest
from fakes import RecordingExecutor, RecordingRepository

from pipeline.generic_services.generic_worker import GenericWorker
from shared.queue_topology import QueueTopology


def _worker(repo: RecordingRepository, adapter: RecordingExecutor) -> GenericWorker:
    return GenericWorker(adapter, QueueTopology("test"), repo, message_broker=None)


def test_unreadable_message_fails_the_task_it_names():
    repo, adapter = RecordingRepository(), RecordingExecutor()
    task_id = str(uuid.uuid4())

    with pytest.raises(KeyError):
        _worker(repo, adapter).handle({"task_id": task_id, "run_name": "r"})

    assert [t for t, _ in repo.failed] == [task_id]
    assert adapter.submitted == []


def test_message_without_a_task_id_is_only_dead_lettered():
    repo, adapter = RecordingRepository(), RecordingExecutor()

    with pytest.raises(KeyError):
        _worker(repo, adapter).handle({"dataset": "wine_quality"})

    assert repo.failed == []
