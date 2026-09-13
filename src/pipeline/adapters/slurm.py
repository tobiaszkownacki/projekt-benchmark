import logging
import os
import shlex

from pipeline.adapters.slurm_connector import SlurmConnector
from pipeline.adapters.slurm_site import SlurmSite
from pipeline.completion import JobState, PollableExecutor
from pipeline.executor import FetchResult, JobDescription, SubmitResult

logger = logging.getLogger(__name__)

SUCCESS_STATE = "COMPLETED"
FAILURE_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
SACCT_CMD = "sacct --parsable2 --allocations --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,End"

# stop_condition keys as a submission stores them, mapped onto the flags
# run_benchmark accepts.
BUDGET_FLAGS = {
    "max_gradient_count": "--max-gradients",
    "max_database_reaches": "--max-db-reaches",
    "max_epochs": "--max-epochs",
}


def _budget_args(stop_condition: dict[str, int]) -> str:
    return " ".join(f"{flag} {stop_condition[key]}" for key, flag in BUDGET_FLAGS.items() if key in stop_condition)


def _parse_sacct(raw: str) -> list[dict]:
    lines = [ln for ln in raw.strip().splitlines() if ln]
    if not lines:
        return []
    header, *rows = lines
    columns = header.split("|")
    return [dict(zip(columns, row.split("|"), strict=True)) for row in rows]


class SlurmExecutor(PollableExecutor):
    """sbatch, sacct and SFTP over SSH. Which centre it talks to is the site."""

    # One SSH connection per call - fine at the poller's 60s cadence.

    def __init__(self, site: SlurmSite) -> None:
        self.site = site
        self.download_dir = os.environ.get("ARTIFACT_ROOT", "/downloads")

    def check_ready(self) -> bool:
        with SlurmConnector(self.site) as cluster:
            partition_state, _, _ = cluster.ssh_capture(f"sinfo -p {self.site.partition} -h -o '%a'")
            if partition_state.strip() != "up":
                logger.warning(f"partition {self.site.partition} not up (state={partition_state!r})")
                return False

            if cluster.account:
                assoc, _, _ = cluster.ssh_capture(
                    f"sacctmgr show assoc user={cluster.user} account={cluster.account} format=Account -n"
                )
                if not assoc.strip():
                    logger.warning(f"no slurm association for user={cluster.user} account={cluster.account}")
                    return False

            reports = f"{cluster.project_dir}/reports"
            _, _, exit_code = cluster.ssh_capture(f"mkdir -p {reports} && test -w {reports}")
            if exit_code != 0:
                logger.warning(f"{reports} is not writable")
                return False
        return True

    def submit_job(self, job: JobDescription) -> SubmitResult:
        with SlurmConnector(self.site) as cluster:
            run_command = (
                f"uv run -m benchmark_core.optimization_engine.run_benchmark "
                f"--dataset {shlex.quote(job.dataset)} --model {shlex.quote(job.model)} "
                f"--optimizer {shlex.quote(job.optimizer)} --seed {job.seed} "
                f"{_budget_args(job.stop_condition)} "
                f"--task-id {shlex.quote(job.task_id)} --plot"
            )
            executor_task_id, stderr = cluster.submit_job(
                job_name=f"job_{job.task_id}",
                run_command=run_command,
                # The engine lives under src/ and is not installed into the
                # project environment, so the module only resolves with src/ on
                # the path.
                env_vars={"PYTHONPATH": f"{cluster.project_dir}/src"},
            )
        logger.info(f"submitted slurm job {executor_task_id} for task_id={job.task_id}")
        return SubmitResult(executor_task_id=executor_task_id, std_out=stderr)

    def fetch_results(self, task_id: str, delete_after_download: bool = False) -> FetchResult:
        local_dir = f"{self.download_dir}/{task_id}"
        with SlurmConnector(self.site) as cluster:
            remote_dir = f"{cluster.project_dir}/reports/task_{task_id}"
            files = cluster.download_results(remote_dir, local_dir)
            if files and delete_after_download:
                cluster.ssh(f"rm -rf {remote_dir}")
                logger.info(f"deleted {remote_dir} on {self.site.name} for task_id={task_id}")
        logger.info(f"downloaded {len(files)} file(s) for task_id={task_id} to {local_dir}")
        return FetchResult(files=files)

    def poll_job_states(self) -> list[JobState]:
        with SlurmConnector(self.site) as cluster:
            rows = _parse_sacct(cluster.ssh(SACCT_CMD))
        return [
            JobState(
                executor_task_id=row.get("JobID", ""),
                job_name=row.get("JobName", ""),
                state=row.get("State", ""),
                succeeded=row.get("State", "") == SUCCESS_STATE,
                failed=row.get("State", "") in FAILURE_STATES,
            )
            for row in rows
        ]

    def fetch_error_tail(self, job_name: str, executor_task_id: str, lines: int = 30) -> str:
        with SlurmConnector(self.site) as cluster:
            remote_out = f"{cluster.project_dir}/reports/{job_name}/{executor_task_id}.out"
            return cluster.ssh(f"tail -n {lines} {remote_out} 2>/dev/null || true")
