import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from pipeline.completion import JobState, PollableExecutor
from pipeline.executor import FetchResult, JobDescription, SubmitResult

logger = logging.getLogger(__name__)

JOB_MODULE = "tools.local_backend.run_job"
JOB_PREFIX = "task_"
JOB_FILE = "job.json"
STATUS_FILE = "status.json"
LOG_FILE = "run.out"

# The executor's own bookkeeping. It stays on this side of the download,
# because everything that crosses it is served to whoever owns the run.
PRIVATE_FILES = {JOB_FILE, STATUS_FILE}

SUCCESS_STATE = "COMPLETED"
FAILURE_STATE = "FAILED"
RUNNING_STATE = "RUNNING"


class LocalExecutor(PollableExecutor):
    """Runs a benchmark on the machine hosting the pipeline.

    A test and CI backend, not a product feature: submit -> poll -> download
    has never run against anything, because cluster credentials are personal
    and everything known about that path came from reading it. The run is
    detached into its own process writing to a staging directory that stands in
    for the cluster's reports tree, and the pipeline learns it finished only by
    polling -- an executor that returned its result from submit_job would prove
    nothing about the path it exists to prove.
    """

    def __init__(self, workspace: str | None = None) -> None:
        self.workspace = Path(workspace or os.environ.get("EXECUTOR_WORKSPACE", "/var/lib/executor-workspace"))
        self.download_dir = Path(os.environ.get("ARTIFACT_ROOT", "/downloads"))

    def _job_dir(self, task_id: str) -> Path:
        # Parsed before it reaches the filesystem, so a malformed id cannot
        # name a directory outside the workspace.
        return self.workspace / f"{JOB_PREFIX}{uuid.UUID(task_id)}"

    def submit_job(self, job: JobDescription) -> SubmitResult:
        job_dir = self._job_dir(job.task_id)
        shutil.rmtree(job_dir, ignore_errors=True)
        job_dir.mkdir(parents=True)

        arguments = [
            sys.executable,
            "-m",
            JOB_MODULE,
            "--job-dir",
            str(job_dir),
            "--dataset",
            job.dataset,
            "--model",
            job.model,
            "--optimizer",
            job.optimizers[0],
            "--seed",
            str(job.seed),
            "--stop-condition",
            json.dumps(job.stop_condition),
        ]
        # Written before the process starts, because a job that leaves no state
        # at all is invisible to the poller, and an invisible job is a task
        # that waits for ever.
        _write_state(job_dir, RUNNING_STATE)
        subprocess.Popen(["sh", "-c", _wrapped(arguments, job_dir)], start_new_session=True)

        executor_task_id = f"local-{uuid.uuid4().hex[:12]}"
        (job_dir / JOB_FILE).write_text(
            json.dumps({"executor_task_id": executor_task_id, "task_id": job.task_id}),
            encoding="utf-8",
        )
        logger.info(f"started local job {executor_task_id} for task_id={job.task_id}")
        return SubmitResult(executor_task_id=executor_task_id, std_out="")

    def fetch_results(self, task_id: str, delete_after_download: bool = False) -> FetchResult:
        job_dir = self._job_dir(task_id)
        local_dir = self.download_dir / str(uuid.UUID(task_id))
        local_dir.mkdir(parents=True, exist_ok=True)

        files: list[str] = []
        for source in sorted(job_dir.rglob("*")):
            if not source.is_file() or source.name in PRIVATE_FILES:
                continue
            target = local_dir / source.relative_to(job_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            files.append(str(target))

        if files and delete_after_download:
            shutil.rmtree(job_dir, ignore_errors=True)

        logger.info(f"copied {len(files)} file(s) for task_id={task_id} to {local_dir}")
        return FetchResult(files=files)

    def poll_job_states(self) -> list[JobState]:
        states: list[JobState] = []
        for job_dir in sorted(self.workspace.glob(f"{JOB_PREFIX}*")):
            job_file = job_dir / JOB_FILE
            status_file = job_dir / STATUS_FILE
            if not job_file.is_file() or not status_file.is_file():
                continue
            bookkeeping = json.loads(job_file.read_text(encoding="utf-8"))
            state = json.loads(status_file.read_text(encoding="utf-8"))["state"]
            states.append(
                JobState(
                    executor_task_id=bookkeeping["executor_task_id"],
                    job_name=job_dir.name,
                    succeeded=state == SUCCESS_STATE,
                    failed=state == FAILURE_STATE,
                    running=state == RUNNING_STATE,
                )
            )
        return states

    def fetch_error_tail(self, job_name: str, executor_task_id: str, lines: int = 30) -> str:
        log = self.workspace / job_name / LOG_FILE
        if not log.is_file():
            return ""
        return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _write_state(job_dir: Path, state: str) -> None:
    (job_dir / STATUS_FILE).write_text(json.dumps({"state": state}), encoding="utf-8")


def _wrapped(arguments: list[str], job_dir: Path) -> str:
    """The run, plus the terminal state it may not live to write itself.

    The same shape as the trap the cluster adapter installs, and for the same
    reason: the process that polls is not the one that started the job, so a
    run that ends without leaving a state is a task nobody ever closes.
    """
    command = " ".join(shlex.quote(argument) for argument in arguments)
    log = shlex.quote(str(job_dir / LOG_FILE))
    status = shlex.quote(str(job_dir / STATUS_FILE))
    return (
        f"{command} > {log} 2>&1; "
        f"if [ $? -eq 0 ]; then state={SUCCESS_STATE}; else state={FAILURE_STATE}; fi; "
        f'printf \'{{"state": "%s"}}\' "$state" > {status}'
    )
