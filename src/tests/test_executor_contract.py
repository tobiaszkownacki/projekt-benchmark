"""One JobDescription, every adapter, the same shape out the other end.

The adapters differ in mechanism -- one submits over SSH, the other runs the job
on this machine -- and nothing else is allowed to differ. These tests are what
catches the drift on the day somebody adds a third one.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.settings import Settings
from pipeline.adapters import local, slurm
from pipeline.completion import JobState, PollableExecutor
from pipeline.executor import ExecutorAdapter, FetchResult, JobDescription, SubmitResult
from pipeline.generic_services.generic_downloader import Downloader
from pipeline.pipeline_builder import _load_from_class_path
from pipeline.pipeline_registration import INFRASTRUCTURES
from pipeline.registered_pipelines import ATHENA
from pipeline.run_result import RunResult, RunSeries, find_manifest
from pipeline.task_repository import TaskRepository, TaskStatus
from shared.queue_topology import QueueTopology

TASK_ID = "6a4f0d38-1f34-4a5e-9d6c-1c2b3a4d5e6f"
EXECUTOR_TASK_ID = "4242"

JOB = JobDescription(
    task_id=TASK_ID,
    dataset="wine",
    model="mlp-1x16",
    optimizer="adam",
    seed=11,
    run_name="adam-wine-s11",
    stop_condition={"max_epochs": 2},
)

RESULT = RunResult(
    stop_reason="EPOCH_LIMIT",
    gradient_count=72,
    database_reaches=2136,
    final_loss=0.0199,
    final_accuracy=97.5,
    total_steps=12,
    total_epochs=2,
    wall_time_seconds=2.2,
    runner_version="local-cpu-1",
    series=RunSeries(
        epochs=[1, 2],
        loss=[0.4, 0.0199],
        accuracy=[80.0, 97.5],
        gradient_count=[36, 72],
        database_reaches=[1068, 2136],
        wall_time_seconds=[1.1, 2.2],
    ),
)

SACCT_OUTPUT = "\n".join(
    [
        "JobID|JobName|Partition|AllocCPUS|State|ExitCode|Elapsed|End",
        f"{EXECUTOR_TASK_ID}|job_{TASK_ID}|plgrid-gpu-a100|1|COMPLETED|0:0|00:00:12|2026-09-13T12:00:00",
    ]
)


def _write_finished_job(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    RESULT.write_manifest(directory)
    (directory / "status.json").write_text(json.dumps({"state": "COMPLETED"}), encoding="utf-8")


class RecordingRepository(TaskRepository):
    def __init__(self) -> None:
        self.results: dict[str, RunResult] = {}
        self.artifacts: dict[str, tuple[int, int]] = {}
        self.errors: dict[str, str] = {}

    def mark_submitted(self, task_id, executor_task_id): ...
    def mark_failed(self, task_id, error_message):
        self.errors[task_id] = error_message

    def set_error(self, task_id, error_message):
        self.errors[task_id] = error_message

    def mark_completed_by_executor_id(self, executor_task_id):
        return True

    def get_by_executor_id(self, executor_task_id):
        return TaskStatus(task_id=TASK_ID, task_status="running")

    def store_result(self, task_id, result):
        self.results[task_id] = result

    def mark_artifacts(self, task_id, files, total_bytes):
        self.artifacts[task_id] = (files, total_bytes)


@pytest.fixture
def slurm_adapter(monkeypatch, tmp_path):
    class FakeConnector:
        project_dir = "/net/people/plgtest/projekt-benchmark"
        account = "plgtest"
        user = "plgtest"

        def __enter__(self):
            return self

        def __exit__(self, *_exception):
            return None

        def submit_job(self, job_name, run_command, env_vars):
            return EXECUTOR_TASK_ID, ""

        def ssh(self, _command):
            return SACCT_OUTPUT

        def ssh_capture(self, _command):
            return "up", "", 0

        def download_results(self, _remote_dir, local_dir):
            _write_finished_job(Path(local_dir))
            return [str(path) for path in sorted(Path(local_dir).iterdir())]

    monkeypatch.setattr(slurm, "SlurmConnector", lambda site: FakeConnector())
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "downloads"))
    return slurm.SlurmExecutor(ATHENA)


@pytest.fixture
def local_adapter(monkeypatch, tmp_path):
    def fake_popen(arguments, **_kwargs):
        _write_finished_job(Path(arguments[arguments.index("--job-dir") + 1]))
        return SimpleNamespace(pid=4242)

    monkeypatch.setattr(local.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "downloads"))
    return local.LocalExecutor(workspace=str(tmp_path / "workspace"))


@pytest.fixture(params=["slurm_adapter", "local_adapter"])
def adapter(request):
    return request.getfixturevalue(request.param)


def test_every_registered_executor_implements_the_interface():
    for name, registration in INFRASTRUCTURES.items():
        executor = _load_from_class_path(registration.executor_path)
        assert issubclass(executor, ExecutorAdapter), f"{name} is not an ExecutorAdapter"
        if registration.completion_path is not None:
            assert issubclass(executor, PollableExecutor), f"{name} has a poller but nothing to poll"


def test_submit_returns_an_executor_task_id(adapter):
    adapter.check_ready()
    submitted = adapter.submit_job(JOB)
    assert isinstance(submitted, SubmitResult)
    assert submitted.executor_task_id


def test_a_finished_job_is_reported_as_one_succeeded_state(adapter):
    adapter.check_ready()
    submitted = adapter.submit_job(JOB)

    states = adapter.poll_job_states()
    assert [type(state) for state in states] == [JobState]
    assert states[0].succeeded is True
    assert states[0].failed is False
    assert states[0].executor_task_id == submitted.executor_task_id


def test_fetching_leaves_the_same_result_manifest(adapter):
    adapter.check_ready()
    adapter.submit_job(JOB)

    fetched = adapter.fetch_results(TASK_ID)
    assert isinstance(fetched, FetchResult)
    manifest = find_manifest(fetched.files)
    assert manifest is not None, "a finished run left no result manifest"
    assert RunResult.from_manifest(manifest) == RESULT


def test_the_downloader_stores_the_same_row_for_every_adapter(adapter):
    adapter.check_ready()
    adapter.submit_job(JOB)

    repository = RecordingRepository()
    downloader = Downloader(adapter, QueueTopology("contract"), repository, message_broker=None)
    downloader.handle({"task_id": TASK_ID})

    assert repository.results[TASK_ID] == RESULT
    files, total_bytes = repository.artifacts[TASK_ID]
    assert files > 0 and total_bytes > 0
    assert not repository.errors


def test_the_default_executor_is_not_the_test_backend():
    # The local adapter is registered so the suite can drive the whole path; a
    # deployment that picked it up by default would run scikit-learn toys and
    # report them as results.
    assert Settings().executor == "athena"
