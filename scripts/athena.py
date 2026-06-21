import os
import re
import time
import paramiko
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ATHENA_HOST     = os.getenv("ATHENA_HOST")
ATHENA_USER     = os.getenv("ATHENA_USER")
ATHENA_PASSWORD = os.getenv("ATHENA_PASSWORD")
ATHENA_ACCOUNT  = os.getenv("ATHENA_ACCOUNT")
POLL_EVERY      = int(os.getenv("POLL_EVERY", 30))

_scratch: str | None = None

def get_client() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        ATHENA_HOST,
        username=ATHENA_USER,
        password=ATHENA_PASSWORD,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def ssh(cmd: str) -> str:
    """Run a command on Athena, return stdout. Raises RuntimeError on non-zero exit."""
    with get_client() as client:
        _, stdout, stderr = client.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode().strip()
        if exit_code != 0:
            raise RuntimeError(stderr.read().decode().strip())
        return out


def get_scratch() -> str:
    """Resolve $SCRATCH once and cache it."""
    global _scratch
    if _scratch is None:
        _scratch = ssh("echo $SCRATCH")
    return _scratch


def generate_job_script(job: dict) -> str:
    """
    Build a SLURM batch script from a flexible job dict.

    Required keys:
        script_remote_path  : absolute remote path to the python script to run

    Optional keys:
        job_name        : str           default "benchmark_job"
        account         : str           default ATHENA_ACCOUNT env var
        time            : str           default "01:00:00"
        mem             : str           default "64G"
        cpus            : int           default 16
        gpus            : int           default 1
        args            : str           extra CLI args passed to the script
        workdir         : str           cd here before running  (default: script's directory)
        extra_sbatch    : list[str]     extra raw #SBATCH lines
        pre_commands    : list[str]     shell commands to run after init, before python
        env_vars        : dict[str,str] exported env vars  e.g. {"TORCH_HOME": "/path"}
    """
    account      = job.get("account", ATHENA_ACCOUNT or "")
    job_name     = job.get("job_name", "benchmark_job")
    time_limit   = job.get("time", "01:00:00")
    mem          = job.get("mem", "64G")
    cpus         = job.get("cpus", 16)
    gpus         = job.get("gpus", 1)
    args         = job.get("args", "")
    script_path  = job.get("script_remote_path")

    if "workdir" in job:
        workdir = job["workdir"]
    elif script_path:
        workdir = str(Path(script_path).parent)
    else:
        workdir = "$SCRATCH"

    run_command = job.get("run_command", f"python {script_path} {args}")

    extra_sbatch  = "\n".join(f"#SBATCH {l}" for l in job.get("extra_sbatch", []))
    env_exports   = "\n".join(f'export {k}="{v}"' for k, v in job.get("env_vars", {}).items())
    pre_commands  = "\n".join(job.get("pre_commands", []))

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time_limit}
#SBATCH --output=%j.out
{extra_sbatch}

{env_exports}

cd {workdir}

{pre_commands}

{run_command}
""".strip()

def upload_file(local_path: str, remote_path: str) -> None:
    """Upload a single file to Athena via SFTP."""
    with get_client() as client:
        with client.open_sftp() as sftp:
            remote_dir = str(Path(remote_path).parent)
            ssh(f"mkdir -p {remote_dir}")
            sftp.put(local_path, remote_path)


def upload_script(local_script: str, job_name: str) -> str:
    """
    Upload a local python script to $SCRATCH/<job_name>/script.py.
    Returns the remote path.
    """
    scratch = get_scratch()
    remote_path = f"{scratch}/{job_name}/script.py"
    upload_file(local_script, remote_path)
    return remote_path

def submit_job(job: dict, local_script: str | None = None) -> str:
    """
    Submit a job to Athena. Returns the Slurm job ID.

    If local_script is provided it is uploaded first and
    script_remote_path is set automatically.

    job dict: see generate_job_script() docstring.
    """
    job = dict(job)

    if local_script:
        job_name = job.get("job_name", "benchmark_job")
        job["script_remote_path"] = upload_script(local_script, job_name)

    script_content = generate_job_script(job)

    scratch    = get_scratch()
    job_name   = job.get("job_name", "benchmark_job")
    job_dir    = f"{scratch}/{job_name}"
    job_sh     = f"{job_dir}/job.sh"

    escaped = script_content.replace("'", "'\\''")
    ssh(f"mkdir -p {job_dir} && printf '%s' '{escaped}' > {job_sh}")

    output = ssh(f"cd {job_dir} && sbatch job.sh")
    match  = re.search(r"(\d+)", output)
    if not match:
        raise RuntimeError(f"Could not parse job ID from sbatch output: {output}")

    job_id = match.group(1)
    print(f"Submitted job {job_id}")
    return job_id


def wait_for_job(job_id: str, poll_every: int = POLL_EVERY) -> str:
    """
    Block until the job leaves the queue. Returns the final Slurm state string.
    Calls on_status(job_id, state) each poll if provided.
    """
    while True:
        try:
            state = ssh(f"squeue -j {job_id} -h -o '%T'")
        except RuntimeError:
            state = ""

        if not state:
            try:
                final = ssh(
                    f"sacct -j {job_id} --noheader -o State | head -1"
                ).split()[0]
            except RuntimeError:
                final = "UNKNOWN"
            print(f"Job {job_id} finished  -  state: {final}")
            return final

        print(f"  Job {job_id}: {state}  -  polling in {poll_every}s")
        time.sleep(poll_every)


def fetch_log(job_id: str, local_dir: str = "./results") -> Path:
    """
    Download the Slurm .out log for job_id to local_dir.
    Returns the local Path of the downloaded file.
    """
    scratch    = get_scratch()
    remote_out = ssh(f"find {scratch} -name '{job_id}.out' | head -1")
    if not remote_out:
        raise FileNotFoundError(f"No .out file found for job {job_id} under {scratch}")

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    local_path = Path(local_dir) / f"{job_id}.out"

    with get_client() as client:
        with client.open_sftp() as sftp:
            sftp.get(remote_out, str(local_path))

    print(f"Fetched log: {local_path}")
    return local_path


def run_job(job: dict, local_script: str | None = None, local_dir: str = "./results") -> dict:
    """
    Full pipeline: upload (optional)  submit  wait  fetch log.
    Returns a result dict with job_id, state, and log_path.
    """
    job_id   = submit_job(job, local_script=local_script)
    state    = wait_for_job(job_id)
    log_path = fetch_log(job_id, local_dir=local_dir)

    return {
        "job_id":   job_id,
        "state":    state,
        "log_path": str(log_path),
        "success":  state == "COMPLETED",
    }