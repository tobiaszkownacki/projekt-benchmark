import logging
import os

from pipeline.adapters.athena_connector import AthenaConnector
from pipeline.completion import CompletionSignal, CompletionSource
from pipeline.executor import ExecutorAdapter, FetchResult, JobDescription, SubmitResult
from pipeline.task_repository import TaskRepository

logger = logging.getLogger(__name__)

SLURM_TIME_LIMIT = "00:30:00"
SLURM_PARTITION = "plgrid-gpu-a100"
MAX_EPOCHS = 10
MAX_GRADIENTS = 100000
FAILURE_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}

SACCT_CMD = (
    "sacct --parsable2 --allocations "
    "--format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,End"
)

ATHENA_REMOTE_PATH = os.environ.get("ATHENA_REMOTE_PATH")
PROJECT_DIR = f"{ATHENA_REMOTE_PATH}/projekt-benchmark"
LOCAL_DOWNLOAD_DIR = os.environ.get("LOCAL_DOWNLOAD_DIR", "/downloads")


def _optimizer_args(optimizers: list[str]) -> str:
    names = [o.strip() for o in optimizers if o.strip()]
    if not names:
        raise ValueError("task has no optimizer selection")
    if len(names) == 1:
        return f"--optimizer {names[0]}"
    return "--compare " + " ".join(names)


def _parse_sacct(raw: str) -> list[dict]:
    lines = [ln for ln in raw.strip().splitlines() if ln]
    if not lines:
        return []
    header, *rows = lines
    columns = header.split("|")
    return [dict(zip(columns, row.split("|"), strict=True)) for row in rows]


class AthenaExecutor(ExecutorAdapter):
    # one SSH connection per call - fine at the poller's 60s cadence

    def check_ready(self) -> bool:
        with AthenaConnector() as athena:
            partition_state, _, _ = athena.ssh_capture(f"sinfo -p {SLURM_PARTITION} -h -o '%a'")
            if partition_state.strip() != "up":
                logger.warning(f"partition {SLURM_PARTITION} not up (state={partition_state!r})")
                return False

            assoc, _, _ = athena.ssh_capture(
                f"sacctmgr show assoc user={athena.user} account={athena.account} format=Account -n"
            )
            if not assoc.strip():
                logger.warning(f"no slurm association for user={athena.user} account={athena.account}")
                return False

            _, _, exit_code = athena.ssh_capture(f"mkdir -p {PROJECT_DIR}/reports && test -w {PROJECT_DIR}/reports")
            if exit_code != 0:
                logger.warning(f"{PROJECT_DIR}/reports is not writable")
                return False
        return True

    def submit_job(self, job: JobDescription) -> SubmitResult:
        slurm_job = {
            "job_name": f"job_{job.task_id}",
            "time": SLURM_TIME_LIMIT,
            "cpus": 1,
            "gpus": 1,
            "workdir": PROJECT_DIR,
            "run_command": (
                f"uv run -m src.benchmark.run_benchmark "
                f"--dataset {job.dataset} {_optimizer_args(job.optimizers)} "
                f"--max-epochs {MAX_EPOCHS} --max-gradients {MAX_GRADIENTS} "
                f"--task-id {job.task_id} --plot"
            ),
        }
        with AthenaConnector() as athena:
            slurm_job_id, stderr = athena.submit_job(slurm_job)
        logger.info(f"submitted slurm job {slurm_job_id} for task_id={job.task_id}")
        return SubmitResult(executor_task_id=slurm_job_id, std_out=stderr)

    def fetch_results(self, task_id: str, delete_after_download: bool = False) -> FetchResult:
        remote_dir = f"{PROJECT_DIR}/reports/task_{task_id}"
        local_dir = f"{LOCAL_DOWNLOAD_DIR}/{task_id}"
        with AthenaConnector() as athena:
            files = athena.download_results(remote_dir, local_dir)
            if files and delete_after_download:
                athena.ssh(f"rm -rf {remote_dir}")
                logger.info(f"deleted {remote_dir} on athena for task_id={task_id}")
        logger.info(f"downloaded {len(files)} file(s) for task_id={task_id} to {local_dir}")
        return FetchResult(files=files)

    def poll_job_states(self) -> list[dict]:
        with AthenaConnector() as athena:
            return _parse_sacct(athena.ssh(SACCT_CMD))

    def fetch_error_tail(self, job_name: str, job_id: str, lines: int = 30) -> str:
        remote_out = f"{PROJECT_DIR}/reports/{job_name}/{job_id}.out"
        with AthenaConnector() as athena:
            return athena.ssh(f"tail -n {lines} {remote_out} 2>/dev/null || true")


class AthenaCompletionSource(CompletionSource):

    def __init__(self, adapter: AthenaExecutor, task_repo: TaskRepository):
        self.adapter = adapter
        self.task_repo = task_repo

    def on_poll(self) -> CompletionSignal:
        if not self.adapter.check_ready():
            logger.info("athena not ready, skipping poll cycle")
            return CompletionSignal([], [])

        completed: list[str] = []
        failed: list[str] = []
        for job in self.adapter.poll_job_states():
            state = job.get("State", "")
            job_id = job.get("JobID", "")
            if state == "COMPLETED":
                task_id = self._reconcile_completed(job_id)
                if task_id:
                    completed.append(task_id)
            elif state in FAILURE_STATES:
                task_id = self._reconcile_failed(job_id, job.get("JobName", ""), state)
                if task_id:
                    failed.append(task_id)
        return CompletionSignal(completed, failed)

    def _reconcile_completed(self, job_id: str) -> str | None:
        record = self.task_repo.get_by_executor_id(job_id)
        if record is None or record.task_status == "completed":
            return None
        if not self.task_repo.mark_completed_by_executor_id(job_id):
            return None
        logger.info(f"task_id={record.task_id} completed (job_id={job_id})")
        return record.task_id

    def _reconcile_failed(self, job_id: str, job_name: str, state: str) -> str | None:
        record = self.task_repo.get_by_executor_id(job_id)
        if record is None:
            logger.warning(f"job_id={job_id} is {state} but has no matching task")
            return None
        if record.task_status == "failed":
            return None
        error_tail = self.adapter.fetch_error_tail(job_name, job_id)
        self.task_repo.mark_failed(record.task_id, error_tail)
        logger.error(f"task_id={record.task_id} failed (job_id={job_id}, state={state})")
        return record.task_id
