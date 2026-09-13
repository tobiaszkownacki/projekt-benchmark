import json
import logging
import os
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

SUCCESS_STATE = "COMPLETED"
FAILURE_STATE = "FAILED"


class LocalExecutor(PollableExecutor):
    """Runs a benchmark on this machine through tools/local_backend.

    A test and CI backend rather than a product feature. It exists because
    submit -> poll -> download has never been executed against anything: nobody
    holds cluster credentials, so everything known about that path came from
    reading it. The run is therefore detached into its own process writing to a
    staging directory, and the pipeline only learns it finished by polling --
    an executor that returned its result from submit_job would exercise none of
    the path it is meant to prove.

    The one thing it does not reproduce: a process killed outright leaves no
    status file, and the task stays pending until someone looks.
    """

    def __init__(self, workspace: str | None = None) -> None:
        self.workspace = Path(workspace or os.environ.get("EXECUTOR_WORKSPACE", "/var/lib/executor-workspace"))
        self.download_dir = Path(os.environ.get("ARTIFACT_ROOT", "/downloads"))

    def check_ready(self) -> bool:
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(f"workspace {self.workspace} is not usable: {exc}")
            return False
        return True

    def _job_dir(self, task_id: str) -> Path:
        # Parsed before it reaches the filesystem, so a malformed id cannot
        # name a directory outside the workspace.
        return self.workspace / f"{JOB_PREFIX}{uuid.UUID(task_id)}"

    def submit_job(self, job: JobDescription) -> SubmitResult:
        job_dir = self._job_dir(job.task_id)
        shutil.rmtree(job_dir, ignore_errors=True)
        job_dir.mkdir(parents=True)

        executor_task_id = f"local-{uuid.uuid4().hex[:12]}"
        (job_dir / JOB_FILE).write_text(
            json.dumps({"executor_task_id": executor_task_id, "task_id": job.task_id}),
            encoding="utf-8",
        )

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
            job.optimizer,
            "--seed",
            str(job.seed),
            "--stop-condition",
            json.dumps(job.stop_condition),
        ]
        with (job_dir / LOG_FILE).open("w", encoding="utf-8") as log:
            subprocess.Popen(arguments, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

        logger.info(f"started local job {executor_task_id} for task_id={job.task_id}")
        return SubmitResult(executor_task_id=executor_task_id, std_out="")

    def fetch_results(self, task_id: str, delete_after_download: bool = False) -> FetchResult:
        job_dir = self._job_dir(task_id)
        local_dir = self.download_dir / str(uuid.UUID(task_id))
        local_dir.mkdir(parents=True, exist_ok=True)

        files: list[str] = []
        for source in sorted(job_dir.rglob("*")):
            if not source.is_file():
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
            executor_task_id = json.loads(job_file.read_text(encoding="utf-8"))["executor_task_id"]
            state = json.loads(status_file.read_text(encoding="utf-8"))["state"]
            states.append(
                JobState(
                    executor_task_id=executor_task_id,
                    job_name=job_dir.name,
                    state=state,
                    succeeded=state == SUCCESS_STATE,
                    failed=state == FAILURE_STATE,
                )
            )
        return states

    def fetch_error_tail(self, job_name: str, executor_task_id: str, lines: int = 30) -> str:
        log = self.workspace / job_name / LOG_FILE
        if not log.is_file():
            return ""
        return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
