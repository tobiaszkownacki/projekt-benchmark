"""The reservation the worker takes before sbatch, against a real database.

It has to stop a second submitter for as long as the first one works, and it
must not outlive the first one: a worker that fails or dies mid-submission may
not leave its task held where no later delivery can submit it.
"""

import threading
import uuid

import pytest


def _reserve_in_another_thread(repository, task_id: str) -> threading.Thread:
    """Starts a second submitter; its answer lands in thread.result once it gets the reservation."""

    def run() -> None:
        with repository.reserve_submission(task_id) as reserved:
            thread.result = reserved

    thread = threading.Thread(target=run)
    thread.result = None
    thread.start()
    return thread


def test_a_second_reservation_waits_and_then_finds_the_job(sql_repository, new_task):
    task_id = new_task()

    with sql_repository.reserve_submission(task_id) as reserved:
        assert reserved
        second = _reserve_in_another_thread(sql_repository, task_id)
        second.join(0.5)
        assert second.is_alive()
        assert sql_repository.mark_submitted(task_id, f"job-{uuid.uuid4().hex}")
    second.join(5)

    assert second.result is False


def test_a_reservation_is_released_when_the_submission_raises(sql_repository, new_task):
    task_id = new_task()

    with pytest.raises(RuntimeError), sql_repository.reserve_submission(task_id):
        raise RuntimeError("sbatch failed")
    second = _reserve_in_another_thread(sql_repository, task_id)
    second.join(5)

    assert second.result is True


@pytest.mark.parametrize(
    ("task_status", "has_job"),
    [("SUBMITTED", True), ("RUNNING", True), ("COMPLETED", True), ("FAILED", False)],
)
def test_only_a_pending_task_without_a_job_is_reserved(sql_repository, new_task, task_status, has_job):
    task_id = new_task(task_status, f"job-{uuid.uuid4().hex}" if has_job else None)

    with sql_repository.reserve_submission(task_id) as reserved:
        assert not reserved
