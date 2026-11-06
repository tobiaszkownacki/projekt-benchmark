"""submit -> poll -> download, end to end, on this machine.

The cluster adapter cannot be exercised anywhere but on the cluster, so this is
the only test there is of the path the whole pipeline is: a job that is started
detached, observed from outside, and only then fetched.

Docker is replaced by fake_docker.py, which records how the container was asked
for and runs its command as a plain process. fake_job_hostile.py plays a job
that tries to mislead or harm the executor. The last test runs the real job
image and is skipped where that image has not been built.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from pipeline.adapters import local
from pipeline.executor import JobDescription
from shared.run_result import RunResult, find_manifest

HERE = Path(__file__).resolve().parent


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{HERE / "fake_docker.py"}" "$@"\n', encoding="utf-8")
    docker.chmod(0o755)
    # The container's command says `python`; here that has to be this interpreter.
    (bin_dir / "python").symlink_to(sys.executable)
    state = tmp_path / "docker-state"
    state.mkdir()
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(state))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", str(HERE))
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "downloads"))
    monkeypatch.setattr(local, "DOCKER", str(docker))
    monkeypatch.setattr(local, "JOB_MODULE", "fake_job")
    yield state
    for record in state.glob("container-*"):
        subprocess.run([str(docker), "rm", "--force", record.name.removeprefix("container-")])


def _executor(tmp_path) -> local.LocalExecutor:
    return local.LocalExecutor(workspace=str(tmp_path / "workspace"))


@pytest.fixture
def executor(tmp_path, fake_docker):
    return _executor(tmp_path)


def _hostile(monkeypatch, mode: str) -> None:
    monkeypatch.setattr(local, "JOB_MODULE", "fake_job_hostile")
    monkeypatch.setenv("FAKE_JOB_MODE", mode)


def _run_calls(state: Path) -> list[list[str]]:
    return json.loads((state / "run_calls.json").read_text(encoding="utf-8"))


def _option(arguments: list[str], name: str) -> list[str]:
    return [arguments[i + 1] for i, argument in enumerate(arguments) if argument == name]


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


def _tail(executor, job, submitted) -> str:
    return executor.fetch_error_tail(f"task_{job.task_id}", submitted.executor_task_id)


# -- how the container is asked for -------------------------------------------


def test_the_job_runs_in_a_sandbox_container(executor, fake_docker, tmp_path):
    job = _job()
    submitted = executor.submit_job(job)

    (arguments,) = _run_calls(fake_docker)
    assert _option(arguments, "--name") == [submitted.executor_task_id]
    assert _option(arguments, "--network") == ["none"]
    assert _option(arguments, "--cap-drop") == ["ALL"]
    assert _option(arguments, "--security-opt") == ["no-new-privileges"]
    assert _option(arguments, "--pull") == ["never"]
    assert "--read-only" in arguments
    assert "--privileged" not in arguments
    assert _option(arguments, "--memory") == _option(arguments, "--memory-swap")
    assert _option(arguments, "--pids-limit") and _option(arguments, "--cpus")
    assert not _option(arguments, "--user")[0].startswith("0:")
    # The job's own work directory is the only mount: no docker socket, no
    # host tree, and not the executor's record of the job.
    work_dir = tmp_path / "workspace" / f"task_{job.task_id}" / "work"
    assert _option(arguments, "--volume") == [f"{work_dir}:/job"]
    image = arguments.index(local.JOB_IMAGE)
    assert _option(arguments, "--entrypoint") == ["timeout"]
    assert arguments[image + 1 : image + 3] == [f"--kill-after={local.KILL_AFTER_SECONDS}", "3600"]


def test_a_root_pipeline_runs_the_job_as_nobody(executor, fake_docker, monkeypatch):
    owners = []
    monkeypatch.setattr(local.os, "getuid", lambda: 0)
    monkeypatch.setattr(local.os, "chown", lambda path, uid, gid: owners.append((Path(path).name, uid, gid)))

    executor.submit_job(_job())

    (arguments,) = _run_calls(fake_docker)
    assert _option(arguments, "--user") == ["65534:65534"]
    assert owners == [("work", 65534, 65534)]


def test_the_work_directory_is_mounted_from_the_host_workspace(tmp_path, monkeypatch, fake_docker):
    monkeypatch.setenv("EXECUTOR_HOST_WORKSPACE", "/srv/executor-workspace")
    job = _job()

    # Only the request matters here; the fake cannot run a job from a host path that does not exist.
    _executor(tmp_path).submit_job(job)

    assert _option(_run_calls(fake_docker)[0], "--volume") == [f"/srv/executor-workspace/task_{job.task_id}/work:/job"]


def test_a_container_docker_refuses_to_start_fails_the_submission(executor, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_REFUSE", "1")

    with pytest.raises(RuntimeError, match="refused by the test"):
        executor.submit_job(_job())

    # Never started, so the poller must not wait for it.
    assert executor.poll_job_states() == []


# -- how a job ends -----------------------------------------------------------


def test_a_finished_job_is_reported_and_fetched_with_its_numbers(executor):
    job = _job()

    submitted = executor.submit_job(job)
    state = _wait_for(executor, lambda s: s.succeeded)

    assert state.executor_task_id == submitted.executor_task_id

    fetched = executor.fetch_results(job.task_id)
    names = {name.rsplit("/", 1)[1] for name in fetched.files}

    assert "result.json" in names
    assert "loss.png" in names
    assert "job.json" not in names
    assert "status.json" not in names

    result = RunResult.from_manifest(find_manifest(fetched.files))
    assert result.final_loss == 0.5
    assert result.series.loss == [0.9, 0.5]


def test_a_job_whose_process_dies_is_reported_failed(executor, monkeypatch):
    monkeypatch.setattr(local, "JOB_MODULE", "fake_job_crash")
    job = _job()

    submitted = executor.submit_job(job)
    state = _wait_for(executor, lambda s: not s.running)

    assert state.failed
    assert "exited with code 3" in _tail(executor, job, submitted)


def test_a_job_cannot_claim_success_by_writing_it(executor, monkeypatch):
    _hostile(monkeypatch, "liar")

    executor.submit_job(_job())
    state = _wait_for(executor, lambda s: not s.running)

    assert state.failed and not state.succeeded


def test_a_job_over_its_time_limit_is_stopped_and_reported_failed(tmp_path, monkeypatch, fake_docker):
    monkeypatch.setattr(local, "JOB_MODULE", "fake_job_slow")
    monkeypatch.setenv("LOCAL_JOB_TIME_LIMIT", "1")
    executor = _executor(tmp_path)
    job = _job()

    submitted = executor.submit_job(job)
    state = _wait_for(executor, lambda s: not s.running)
    time.sleep(2)

    assert state.failed
    assert "time limit" in _tail(executor, job, submitted)
    assert not (tmp_path / "workspace" / f"task_{job.task_id}" / "work" / "finished").exists()


def test_a_job_that_ignores_the_time_limit_is_killed(tmp_path, monkeypatch, fake_docker):
    _hostile(monkeypatch, "stubborn")
    monkeypatch.setattr(local, "KILL_AFTER_SECONDS", 1)
    monkeypatch.setenv("LOCAL_JOB_TIME_LIMIT", "1")
    executor = _executor(tmp_path)

    executor.submit_job(_job())
    state = _wait_for(executor, lambda s: not s.running, timeout=15)

    assert state.failed


def test_a_job_past_its_deadline_is_removed_by_the_poller(executor, monkeypatch, tmp_path):
    _hostile(monkeypatch, "stubborn")
    job = _job()
    submitted = executor.submit_job(job)
    record_file = tmp_path / "workspace" / f"task_{job.task_id}" / "job.json"
    record = json.loads(record_file.read_text())
    record["started_at"] -= 10 * 3600
    record_file.write_text(json.dumps(record))

    state = _wait_for(executor, lambda s: not s.running)

    assert state.failed
    assert "time limit" in _tail(executor, job, submitted)


def test_a_job_that_writes_too_much_is_stopped(tmp_path, monkeypatch, fake_docker):
    _hostile(monkeypatch, "flood")
    monkeypatch.setenv("LOCAL_JOB_MAX_OUTPUT_MB", "1")
    executor = _executor(tmp_path)
    job = _job()

    submitted = executor.submit_job(job)
    state = _wait_for(executor, lambda s: not s.running)

    assert state.failed
    assert "more than 1 MB" in _tail(executor, job, submitted)
    assert executor.fetch_results(job.task_id).files == []


def test_a_cancelled_job_stops_and_is_reported_failed(executor, monkeypatch, tmp_path):
    monkeypatch.setattr(local, "JOB_MODULE", "fake_job_slow")
    job = _job()

    submitted = executor.submit_job(job)
    executor.cancel_job(submitted.executor_task_id)
    state = _wait_for(executor, lambda s: not s.running)
    time.sleep(3)

    assert state.failed
    assert not (tmp_path / "workspace" / f"task_{job.task_id}" / "work" / "finished").exists()


def test_a_job_whose_container_vanished_is_reported_failed(executor, monkeypatch):
    monkeypatch.setattr(local, "JOB_MODULE", "fake_job_slow")
    job = _job()
    submitted = executor.submit_job(job)

    # Gone without a word, as after a Docker reset.
    subprocess.run([local.DOCKER, "rm", "--force", submitted.executor_task_id], check=True)
    state = _wait_for(executor, lambda s: not s.running)

    assert state.failed
    assert "disappeared" in _tail(executor, job, submitted)


def test_a_finished_job_stays_finished_after_its_container_is_removed(executor):
    job = _job()
    executor.submit_job(job)
    _wait_for(executor, lambda s: s.succeeded)

    executor.fetch_results(job.task_id)  # removes the container

    assert executor.poll_job_states()[0].succeeded


# -- what the host reads from the job's directory -----------------------------


def test_files_are_not_fetched_while_the_job_runs(executor, monkeypatch):
    monkeypatch.setattr(local, "JOB_MODULE", "fake_job_slow")
    job = _job()
    executor.submit_job(job)

    with pytest.raises(RuntimeError, match="has not finished"):
        executor.fetch_results(job.task_id)


def test_links_planted_by_a_job_are_never_followed(executor, monkeypatch, tmp_path):
    secret = tmp_path / "host" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text("HOST SECRET", encoding="utf-8")
    _hostile(monkeypatch, "links")
    monkeypatch.setenv("FAKE_JOB_HOST_FILE", str(secret))
    job = _job()

    submitted = executor.submit_job(job)
    state = _wait_for(executor, lambda s: not s.running)
    tail = _tail(executor, job, submitted)
    fetched = executor.fetch_results(job.task_id)
    executor.cancel_job(submitted.executor_task_id)

    assert state.failed
    assert "a normal log line" in tail
    assert "HOST SECRET" not in tail
    assert all("HOST SECRET" not in Path(name).read_text(errors="replace") for name in fetched.files)
    # Nothing was written through the links either.
    assert secret.read_text(encoding="utf-8") == "HOST SECRET"


def test_a_fifo_left_by_a_job_does_not_hang_the_download(executor, monkeypatch):
    _hostile(monkeypatch, "fifo")
    job = _job()
    executor.submit_job(job)
    _wait_for(executor, lambda s: s.succeeded)

    fetched = executor.fetch_results(job.task_id)

    assert [Path(name).name for name in fetched.files] == ["after.txt"]


# -- the real image -----------------------------------------------------------


def _real_image_available() -> bool:
    if not shutil.which("docker"):
        return False
    probe = subprocess.run(["docker", "image", "inspect", local.JOB_IMAGE], capture_output=True)
    return probe.returncode == 0


@pytest.mark.skipif(not _real_image_available(), reason=f"the job image {local.JOB_IMAGE} is not built here")
def test_a_real_job_runs_in_the_job_image(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "downloads"))
    executor = _executor(tmp_path)
    job = _job(dataset="wine", model="mlp-2x32", optimizers=["adam"], stop_condition={"max_epochs": 2})

    submitted = executor.submit_job(job)
    state = _wait_for(executor, lambda s: not s.running, timeout=300)

    assert state.succeeded, _tail(executor, job, submitted)
    result = RunResult.from_manifest(find_manifest(executor.fetch_results(job.task_id).files))
    assert result.total_epochs == 2
