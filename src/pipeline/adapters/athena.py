import logging
import os
import shlex

from pipeline.adapters.athena_connector import AthenaConnector
from pipeline.completion import JobState, PollableExecutor
from pipeline.executor import FetchResult, JobDescription, SubmitResult

logger = logging.getLogger(__name__)

SLURM_TIME_LIMIT = os.environ.get("ATHENA_TIME_LIMIT", "00:30:00")
SUCCESS_STATE = "COMPLETED"
RUNNING_STATE = "RUNNING"
FAILURE_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}

SACCT_CMD = "sacct --parsable2 --allocations --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,End"

ATHENA_REMOTE_PATH = os.environ.get("ATHENA_REMOTE_PATH")
PROJECT_DIR = f"{ATHENA_REMOTE_PATH}/projekt-benchmark"
LOCAL_DOWNLOAD_DIR = os.environ.get("LOCAL_DOWNLOAD_DIR", "/downloads")
# uv is not installed on the compute nodes, so the job finds it where it was
# installed into the account.
ATHENA_PATH_PREFIX = os.environ.get("ATHENA_PATH_PREFIX", "$SCRATCH/.local/bin")
ATHENA_WEBHOOK_URL = os.environ.get("ATHENA_WEBHOOK_URL", "")
ATHENA_LOGIN_NODE = os.environ.get("ATHENA_LOGIN_NODE", "athena.cyfronet.pl")


def _webhook_trap(webhook_token: str) -> str:
    """Report the run's own end on the way out.

    Posted from the login node over ssh because compute nodes have no route
    off the cluster. --max-time keeps a hanging call from holding the node
    after the run is over, and retrying is safe: the receiver applies the
    first report and answers the rest with applied=false.
    """
    if not ATHENA_WEBHOOK_URL:
        raise RuntimeError("ATHENA_WEBHOOK_URL is not set, so a job would report its end into nothing")

    callback_url = f"{ATHENA_WEBHOOK_URL}?job_id=$SLURM_JOB_ID&state=$FINAL_STATE&exit_code=$EXIT_CODE"
    curl_cmd = (
        f"curl -sS -X POST --max-time 10 --retry 2 --retry-connrefused "
        f'-H \\"Authorization: Bearer {webhook_token}\\" \\"{callback_url}\\"'
    )
    return f'''trap '
  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 0 ]; then FINAL_STATE="COMPLETED"; else FINAL_STATE="FAILED"; fi
  ssh -o StrictHostKeyChecking=no {ATHENA_LOGIN_NODE} "{curl_cmd}"
' EXIT SIGTERM'''


# stop_condition keys as a submission stores them, mapped onto the flags
# run_benchmark accepts.
BUDGET_FLAGS = {
    "max_epochs": "--max-epochs",
    "max_gradient_count": "--max-gradients",
    "max_database_reaches": "--max-db-reaches",
}


def _budget_args(stop_condition: dict[str, int]) -> str:
    return " ".join(f"{flag} {stop_condition[key]}" for key, flag in BUDGET_FLAGS.items() if key in stop_condition)


def _optimizer_args(optimizers: list[str]) -> str:
    names = [o.strip() for o in optimizers if o.strip()]
    if not names:
        raise ValueError("task has no optimizer selection")
    if len(names) == 1:
        return f"--optimizer {shlex.quote(names[0])}"
    return "--compare " + " ".join(shlex.quote(name) for name in names)


def _parse_sacct(raw: str) -> list[dict]:
    lines = [ln for ln in raw.strip().splitlines() if ln]
    if not lines:
        return []
    header, *rows = lines
    columns = header.split("|")
    return [dict(zip(columns, row.split("|"), strict=True)) for row in rows]


class AthenaExecutor(PollableExecutor):
    # one SSH connection per call - fine at the poller's 60s cadence

    def submit_job(self, job: JobDescription) -> SubmitResult:
        slurm_job = {
            # The batch script writes its stdout to reports/<job_name>/, and
            # the download reads reports/task_<task_id>/. Naming the job after
            # the task makes those the same directory, so the run's own log is
            # an artifact rather than something only a failure ever shows.
            "job_name": f"task_{job.task_id}",
            "time": SLURM_TIME_LIMIT,
            "cpus": 1,
            "gpus": 1,
            "workdir": PROJECT_DIR,
            "run_command": (
                f"uv run -m benchmark_core.optimization_engine.run_benchmark "
                f"--dataset {shlex.quote(job.dataset)} --model {shlex.quote(job.model)} "
                f"{_optimizer_args(job.optimizers)} --seed {job.seed} "
                f"{_budget_args(job.stop_condition)} "
                f"--task-id {shlex.quote(job.task_id)} --plot"
            ),
            # The engine lives under src/ and is not installed into the project
            # environment, so the module resolves only with src/ on the path.
            "env_vars": {"PYTHONPATH": f"{PROJECT_DIR}/src", "PATH": f"{ATHENA_PATH_PREFIX}:$PATH"},
        }
        if job.webhook_token:
            slurm_job["pre_commands"] = [_webhook_trap(job.webhook_token)]
        with AthenaConnector() as athena:
            slurm_job_id, stderr = athena.submit_job(slurm_job)
        logger.info(f"submitted slurm job {slurm_job_id} for task_id={job.task_id}")
        return SubmitResult(executor_task_id=slurm_job_id, std_out=stderr)

    def fetch_results(self, task_id: str, delete_after_download: bool = False) -> FetchResult:
        """
        method for downloading
        """
        remote_dir = f"{PROJECT_DIR}/reports/task_{task_id}"
        local_dir = f"{LOCAL_DOWNLOAD_DIR}/{task_id}"
        with AthenaConnector() as athena:
            files = athena.download_results(remote_dir, local_dir)
            if files and delete_after_download:
                athena.ssh(f"rm -rf {remote_dir}")
                logger.info(f"deleted {remote_dir} on athena for task_id={task_id}")
        logger.info(f"downloaded {len(files)} file(s) for task_id={task_id} to {local_dir}")
        return FetchResult(files=files)

    def poll_job_states(self) -> list[JobState]:
        with AthenaConnector() as athena:
            rows = _parse_sacct(athena.ssh(SACCT_CMD))
        return [
            JobState(
                executor_task_id=row.get("JobID", ""),
                job_name=row.get("JobName", ""),
                succeeded=row.get("State", "") == SUCCESS_STATE,
                failed=row.get("State", "") in FAILURE_STATES,
                running=row.get("State", "") == RUNNING_STATE,
            )
            for row in rows
        ]

    def fetch_error_tail(self, job_name: str, executor_task_id: str, lines: int = 30) -> str:
        remote_out = f"{PROJECT_DIR}/reports/{job_name}/{executor_task_id}.out"
        with AthenaConnector() as athena:
            return athena.ssh(f"tail -n {lines} {remote_out} 2>/dev/null || true")
