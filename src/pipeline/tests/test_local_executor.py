"""submit -> poll -> download, end to end, on this machine.

The cluster adapter cannot be exercised anywhere but on the cluster, so this is
the only test there is of the path the whole pipeline is: a job that is started
detached, observed through the filesystem, and only then fetched.
"""

import time
import uuid

import pytest

from pipeline.adapters import local
from pipeline.executor import JobDescription
from shared.run_result import RunResult, find_manifest

HERE = __file__.rsplit("/", 1)[0]


@pytest.fixture
def executor(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "JOB_MODULE", "fake_job")
    monkeypatch.setenv("PYTHONPATH", HERE)
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "downloads"))
    return local.LocalExecutor(workspace=str(tmp_path / "workspace"))


def _job(**overrides) -> JobDescription:
    fields = {
        "task_id": str(uuid.uuid4()),
        "dataset": "wine_quality",
        "model": "mlp-2x32",
        "optimizers": ["adam"],
        "seed": 11,
        "run_name": "run",
        "stop_condition": {"max_epochs": 2},
    }
    fields.update(overrides)
    return JobDescription(**fields)


def _wait_for(executor, predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        states = executor.poll_job_states()
        if states and predicate(states[0]):
            return states[0]
        time.sleep(0.1)
    raise AssertionError(f"job never reached the expected state: {executor.poll_job_states()}")


def test_a_finished_job_is_reported_and_fetched_with_its_numbers(executor):
    job = _job()

    submitted = executor.submit_job(job)
    state = _wait_for(executor, lambda s: s.succeeded)

    assert state.executor_task_id == submitted.executor_task_id

    fetched = executor.fetch_results(job.task_id)
    names = {name.rsplit("/", 1)[1] for name in fetched.files}

    assert "result.json" in names
    assert "loss.png" in names
    # Bookkeeping the participant has no business reading.
    assert "job.json" not in names
    assert "status.json" not in names

    result = RunResult.from_manifest(find_manifest(fetched.files))
    assert result.final_loss == 0.5
    assert result.series.loss == [0.9, 0.5]


def test_a_job_whose_process_dies_is_reported_failed(executor, monkeypatch):
    monkeypatch.setattr(local, "JOB_MODULE", "fake_job_crash")
    job = _job()

    executor.submit_job(job)
    state = _wait_for(executor, lambda s: not s.running)

    assert state.failed
    assert not state.succeeded
