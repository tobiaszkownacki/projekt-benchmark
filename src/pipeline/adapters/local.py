import json
import logging
import os
import shutil
import stat
import subprocess
import time
import uuid
from pathlib import Path

from pipeline.completion import JobState, PollableExecutor
from pipeline.executor import FetchResult, JobDescription, SubmitResult

logger = logging.getLogger(__name__)

JOB_MODULE = "tools.local_backend.run_job"
JOB_PREFIX = "task_"
# The executor's own record of a job. It sits next to the job's directory,
# never inside it: the host does not open, follow or write anything the job
# could have placed there, apart from copying out regular files once the job
# has stopped.
JOB_FILE = "job.json"
# Everything the job may write, mounted into its container at /job.
WORK_DIR = "work"
CONTAINER_JOB_DIR = "/job"

SUCCESS_STATE = "COMPLETED"
FAILURE_STATE = "FAILED"
RUNNING_STATE = "RUNNING"

DOCKER = "docker"
JOB_IMAGE = os.environ.get("LOCAL_JOB_IMAGE", "benchmark-local-job:latest")
JOB_LABEL = "benchmark.local-job"
# The user a job runs as when the pipeline itself runs as root.
NOBODY = 65534
# How long a job may ignore the end of its time limit before it is killed.
KILL_AFTER_SECONDS = 30
# Bounds on what one job may leave behind for the download.
MAX_FILES = 10_000


class LocalExecutor(PollableExecutor):
    """Runs a benchmark on the machine hosting the pipeline, inside a sandbox container.

    Every job gets a fresh container from the job image with no network, no
    Linux capabilities, a read-only filesystem, an unprivileged user and limits
    on memory, CPU, processes, wall time and output size. The only place it can
    write is its own work directory, mounted at /job. The optimizer code never
    runs on the host itself.

    Whether a job is running, and how it ended, comes from Docker: the
    container's state and exit code, and its log. Nothing the job writes is
    trusted for that. The container is detached and the pipeline learns that it
    finished only by polling, the same way it learns it from the cluster.
    """

    def __init__(self, workspace: str | None = None) -> None:
        self.workspace = Path(workspace or os.environ.get("EXECUTOR_WORKSPACE", "/var/lib/executor-workspace"))
        # Docker mounts paths of the host. When the pipeline runs in a
        # container, set this to where the workspace lives on the host.
        self.host_workspace = Path(os.environ.get("EXECUTOR_HOST_WORKSPACE", str(self.workspace)))
        self.download_dir = Path(os.environ.get("ARTIFACT_ROOT", "/downloads"))
        self.memory = os.environ.get("LOCAL_JOB_MEMORY", "4g")
        self.cpus = os.environ.get("LOCAL_JOB_CPUS", "2")
        self.time_limit = int(os.environ.get("LOCAL_JOB_TIME_LIMIT", "3600"))
        self.max_output_bytes = int(os.environ.get("LOCAL_JOB_MAX_OUTPUT_MB", "1024")) * 1024 * 1024

    def _job_dir(self, task_id: str) -> Path:
        # Parsed before it reaches the filesystem, so a malformed id cannot
        # name a directory outside the workspace.
        return self.workspace / f"{JOB_PREFIX}{uuid.UUID(task_id)}"

    # -- submit ---------------------------------------------------------------

    def submit_job(self, job: JobDescription) -> SubmitResult:
        job_dir = self._job_dir(job.task_id)
        shutil.rmtree(job_dir, ignore_errors=True)
        work_dir = job_dir / WORK_DIR
        work_dir.mkdir(parents=True)
        user = _hand_over(work_dir)

        name = f"local-{uuid.uuid4().hex[:12]}"
        try:
            started = _docker("run", *self._run_arguments(job, work_dir, name, user))
        except subprocess.TimeoutExpired as error:
            # The daemon may still create the container; do not leave it running untracked.
            _docker_quietly("rm", "--force", name)
            raise RuntimeError(f"starting the job container timed out: {error}") from error
        if started.returncode != 0:
            raise RuntimeError(f"could not start the job container: {started.stderr.strip()}")

        # Written only once the container exists, so the poller never waits
        # for a job that was not started.
        _write_record(
            job_dir,
            {"executor_task_id": name, "task_id": job.task_id, "started_at": time.time()},
        )
        logger.info(f"started local job {name} for task_id={job.task_id}")
        return SubmitResult(executor_task_id=name, std_out=started.stdout.strip())

    def _run_arguments(self, job: JobDescription, work_dir: Path, name: str, user: str) -> list[str]:
        host_work_dir = self.host_workspace / work_dir.relative_to(self.workspace)
        return [
            "--detach",
            "--pull",
            "never",
            "--name",
            name,
            "--label",
            f"{JOB_LABEL}={job.task_id}",
            "--user",
            user,
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=512m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--memory",
            self.memory,
            "--memory-swap",
            self.memory,
            "--cpus",
            self.cpus,
            # The log lives on the host's disk, outside the output limit.
            "--log-opt",
            "max-size=10m",
            "--log-opt",
            "max-file=1",
            "--env",
            "HOME=/tmp",
            "--env",
            "MPLCONFIGDIR=/tmp",
            "--volume",
            f"{host_work_dir}:{CONTAINER_JOB_DIR}",
            # timeout ends the run at its limit and kills it if it ignores that.
            "--entrypoint",
            "timeout",
            JOB_IMAGE,
            f"--kill-after={KILL_AFTER_SECONDS}",
            str(self.time_limit),
            "python",
            "-m",
            JOB_MODULE,
            "--job-dir",
            CONTAINER_JOB_DIR,
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

    # -- poll -----------------------------------------------------------------

    def poll_job_states(self) -> list[JobState]:
        containers = _job_containers()
        states: list[JobState] = []
        for job_dir in sorted(self.workspace.glob(f"{JOB_PREFIX}*")):
            record = _read_record(job_dir)
            if record is None:
                continue
            if record.get("state") not in (SUCCESS_STATE, FAILURE_STATE) and containers is not None:
                record = self._settle(job_dir, record, containers.get(record["executor_task_id"]))
            state = record.get("state", RUNNING_STATE)
            states.append(
                JobState(
                    executor_task_id=record["executor_task_id"],
                    job_name=job_dir.name,
                    succeeded=state == SUCCESS_STATE,
                    failed=state == FAILURE_STATE,
                    running=state == RUNNING_STATE,
                )
            )
        return states

    def _settle(self, job_dir: Path, record: dict, container_state: str | None) -> dict:
        """Records how a job ended, once Docker says it has; a running job is checked against its limits."""
        name = record["executor_task_id"]
        if container_state is None:
            # Gone without an exit code: removed from outside, a Docker reset, a reboot.
            return _finish(job_dir, record, FAILURE_STATE, "The job container disappeared before it finished.")
        if container_state in ("exited", "dead"):
            exit_code, oom_killed = _exit_status(name)
            if exit_code == 0 and not oom_killed:
                return _finish(job_dir, record, SUCCESS_STATE, "")
            if oom_killed:
                reason = f"The job ran out of memory (limit {self.memory})."
            elif exit_code in (124, 137):
                reason = f"The job exceeded its time limit of {self.time_limit} s."
            else:
                reason = f"The job exited with code {exit_code}."
            return _finish(job_dir, record, FAILURE_STATE, reason)

        # Still running (or created): enforce what the container itself cannot.
        # A job is reported failed only once its container is really gone;
        # otherwise the next poll tries again.
        if time.time() - record["started_at"] > self.time_limit + KILL_AFTER_SECONDS + 60 and _remove(name):
            return _finish(job_dir, record, FAILURE_STATE, f"The job exceeded its time limit of {self.time_limit} s.")
        if _tree_size(job_dir / WORK_DIR, self.max_output_bytes) > self.max_output_bytes and _remove(name):
            limit_mb = self.max_output_bytes // (1024 * 1024)
            return _finish(job_dir, record, FAILURE_STATE, f"The job wrote more than {limit_mb} MB.")
        return record

    def fetch_error_tail(self, job_name: str, executor_task_id: str, lines: int = 30) -> str:
        record = _read_record(self.workspace / job_name) or {}
        reason = record.get("reason", "")
        logs = _docker_quietly("logs", "--tail", str(lines), executor_task_id)
        tail = ""
        if logs is not None and logs.returncode == 0:
            # The job's own output, taken from Docker, never from a file it controls.
            tail = (logs.stdout + logs.stderr)[-16_000:].strip()
        return "\n".join(part for part in (reason, tail) if part)

    # -- download -------------------------------------------------------------

    def fetch_results(self, task_id: str, delete_after_download: bool = False) -> FetchResult:
        job_dir = self._job_dir(task_id)
        record = _read_record(job_dir)
        if record is None or record.get("state") not in (SUCCESS_STATE, FAILURE_STATE):
            # A running job could still swap files under the copy.
            raise RuntimeError(f"task_id={task_id} has not finished, so its files are not fetched yet")

        local_dir = self.download_dir / str(uuid.UUID(task_id))
        local_dir.mkdir(parents=True, exist_ok=True)
        files = _copy_regular_files(job_dir / WORK_DIR, local_dir, self.max_output_bytes)

        # The container has stopped; its log has been read by now.
        _docker_quietly("rm", "--force", record["executor_task_id"])
        if files and delete_after_download:
            shutil.rmtree(job_dir, ignore_errors=True)

        logger.info(f"copied {len(files)} file(s) for task_id={task_id} to {local_dir}")
        return FetchResult(files=files)

    def cancel_job(self, executor_task_id: str) -> None:
        """Ends the run and reports it failed, as a cancelled job on the cluster is."""
        for job_dir in self.workspace.glob(f"{JOB_PREFIX}*"):
            record = _read_record(job_dir)
            if record is None or record["executor_task_id"] != executor_task_id:
                continue
            if not _remove(executor_task_id):
                raise RuntimeError(f"could not stop local job {executor_task_id}")
            _finish(job_dir, record, FAILURE_STATE, "The job was cancelled.")
            logger.info(f"cancelled local job {executor_task_id}")
            return
        logger.warning(f"local job {executor_task_id} is not in {self.workspace}, so nothing was cancelled")


# -- Docker -------------------------------------------------------------------


def _docker(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run([DOCKER, *arguments], capture_output=True, text=True, timeout=60)


def _docker_quietly(*arguments: str) -> subprocess.CompletedProcess | None:
    try:
        return _docker(*arguments)
    except (OSError, subprocess.TimeoutExpired) as error:
        logger.warning(f"docker {arguments[0]} failed: {error}")
        return None


def _remove(name: str) -> bool:
    """Stops and removes a job container; True once it no longer exists."""
    removed = _docker_quietly("rm", "--force", name)
    if removed is None:
        return False
    return removed.returncode == 0 or "No such container" in removed.stderr


def _job_containers() -> dict[str, str] | None:
    """State of every job container by name, or None when Docker cannot say.

    None leaves every job as it was: a daemon that does not answer is not a
    job that ended.
    """
    listed = _docker_quietly("ps", "--all", "--filter", f"label={JOB_LABEL}", "--format", "{{.Names}} {{.State}}")
    if listed is None or listed.returncode != 0:
        logger.warning("could not list job containers")
        return None
    containers = {}
    for line in listed.stdout.splitlines():
        name, _, state = line.strip().partition(" ")
        if name:
            containers[name] = state
    return containers


def _exit_status(name: str) -> tuple[int, bool]:
    inspected = _docker_quietly("inspect", "--format", "{{.State.ExitCode}} {{.State.OOMKilled}}", name)
    if inspected is None or inspected.returncode != 0:
        return -1, False
    code, _, oom = inspected.stdout.strip().partition(" ")
    try:
        return int(code), oom == "true"
    except ValueError:
        return -1, False


# -- the executor's own record ------------------------------------------------


def _read_record(job_dir: Path) -> dict | None:
    try:
        return json.loads((job_dir / JOB_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_record(job_dir: Path, record: dict) -> None:
    staged = job_dir / f".{JOB_FILE}.{os.getpid()}"
    staged.write_text(json.dumps(record), encoding="utf-8")
    os.replace(staged, job_dir / JOB_FILE)


def _finish(job_dir: Path, record: dict, state: str, reason: str) -> dict:
    if state == FAILURE_STATE:
        logger.warning(f"local job {record['executor_task_id']} failed: {reason}")
    record = {**record, "state": state, "reason": reason}
    _write_record(job_dir, record)
    return record


# -- the job's directory: read without following anything it placed there ---


def _hand_over(work_dir: Path) -> str:
    """Gives the work directory to the unprivileged user the job runs as, and names that user.

    Never root: under a root pipeline the job runs as nobody, otherwise as the
    user running the pipeline.
    """
    if os.getuid() != 0:
        return f"{os.getuid()}:{os.getgid()}"
    os.chown(work_dir, NOBODY, NOBODY)
    return f"{NOBODY}:{NOBODY}"


def _tree_size(root: Path, stop_above: int) -> int:
    total = 0
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            try:
                total += os.lstat(os.path.join(directory, name)).st_size
            except OSError:
                continue
            if total > stop_above:
                return total
    return total


def _copy_regular_files(source_root: Path, target_root: Path, max_bytes: int) -> list[str]:
    """Copies the regular files a stopped job left, never following a link, within the output limits."""
    files: list[str] = []
    copied = 0
    for directory, subdirectories, names in os.walk(source_root, followlinks=False):
        subdirectories.sort()
        for name in sorted(names):
            source = Path(directory) / name
            relative = source.relative_to(source_root)
            if relative.parts[0] == "status.json" and len(relative.parts) == 1:
                continue  # the runner's own marker, meaningless to the participant
            if len(files) >= MAX_FILES:
                logger.warning(f"{source_root}: more than {MAX_FILES} files, the rest is not fetched")
                return files
            try:
                # O_NONBLOCK: opening a FIFO the job left would otherwise wait for ever.
                descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            except OSError:
                continue  # a symlink, or gone
            with os.fdopen(descriptor, "rb") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode):
                    continue
                if copied + info.st_size > max_bytes:
                    logger.warning(f"{source_root}: output over {max_bytes} bytes, the rest is not fetched")
                    return files
                target = target_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "wb") as out:
                    shutil.copyfileobj(handle, out)
                copied += info.st_size
                files.append(str(target))
    return files
