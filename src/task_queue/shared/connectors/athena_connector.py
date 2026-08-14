import os
import re
import stat
import time
import paramiko
from pathlib import Path

POLL_EVERY = int(os.environ.get("POLL_EVERY", 30))


class AthenaConnector:

    def __init__(self) -> None:
        self.host = os.environ.get("ATHENA_HOST")
        self.user = os.environ.get("ATHENA_USER")
        self.password = os.environ.get("ATHENA_PASSWORD")
        self.account = os.environ.get("ATHENA_ACCOUNT")
        self.client = None
        self._scratch: str | None = None

    def __enter__(self) -> "AthenaConnector":
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            self.host,
            username=self.user,
            password=self.password,
            look_for_keys=False,
            allow_agent=False,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def ssh(self, cmd: str) -> str:
        """Run a command on Athena, return stdout. Raises RuntimeError on non-zero exit."""
        _, stdout, stderr = self.client.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode().strip()
        if exit_code != 0:
            raise RuntimeError(stderr.read().decode().strip())
        return out

    def ssh_capture(self, cmd: str) -> tuple[str, str, int]:
        """Run a command on Athena, returning (stdout, stderr, exit_code) without raising."""
        _, stdout, stderr = self.client.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        return stdout.read().decode().strip(), stderr.read().decode().strip(), exit_code

    def get_scratch(self) -> str:
        """Resolve $SCRATCH once and cache it."""
        if self._scratch is None:
            self._scratch = self.ssh("echo $SCRATCH")
        return self._scratch

    def generate_job_script(self, job: dict) -> str:
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
            pre_commands    : list[str]     shell commands to run after db_schemas, before python
            env_vars        : dict[str,str] exported env vars  e.g. {"TORCH_HOME": "/path"}
        """
        account      = job.get("account", self.account or "")
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
#SBATCH --output={workdir}/reports/{job_name}/%j.out
{extra_sbatch}

{env_exports}

cd {workdir}

{pre_commands}

{run_command}
""".strip()

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a single file to Athena via SFTP."""
        remote_dir = str(Path(remote_path).parent)
        self.ssh(f"mkdir -p {remote_dir}")
        with self.client.open_sftp() as sftp:
            sftp.put(local_path, remote_path)

    def upload_script(self, local_script: str, job_name: str) -> str:
        """
        Upload a local python script to $SCRATCH/<job_name>/script.py.
        Returns the remote path.
        """
        scratch = self.get_scratch()
        remote_path = f"{scratch}/{job_name}/script.py"
        self.upload_file(local_script, remote_path)
        return remote_path

    def submit_job(self, job: dict, local_script: str | None = None) -> tuple[str, str]:
        """
        Submit a job to Athena. Returns (job_id, stderr).

        stderr is whatever the sbatch command wrote to stderr -- it may be
        non-empty even on a successful submission (e.g. warnings), so callers
        should inspect it even when no exception is raised.

        If local_script is provided it is uploaded first and
        script_remote_path is set automatically.

        job dict: see generate_job_script() docstring.
        """
        job = dict(job)

        if local_script:
            job_name = job.get("job_name", "benchmark_job")
            job["script_remote_path"] = self.upload_script(local_script, job_name)

        script_content = self.generate_job_script(job)

        scratch    = self.get_scratch()
        job_name   = job.get("job_name", "benchmark_job")
        workdir    = job.get("workdir", "$SCRATCH")
        job_dir    = f"{scratch}/{job_name}"
        job_sh     = f"{job_dir}/job.sh"

        # the output dir must exist before sbatch starts, since Slurm won't create it for --output
        self.ssh(f"mkdir -p {workdir}/reports/{job_name}")

        escaped = script_content.replace("'", "'\\''")
        self.ssh(f"mkdir -p {job_dir} && printf '%s' '{escaped}' > {job_sh}")

        stdout, stderr, exit_code = self.ssh_capture(f"cd {job_dir} && sbatch job.sh")
        if exit_code != 0:
            raise RuntimeError(stderr or f"sbatch exited with status {exit_code}")

        match = re.search(r"(\d+)", stdout)
        if not match:
            raise RuntimeError(f"Could not parse job ID from sbatch output: {stdout!r}")

        return match.group(1), stderr

    def wait_for_job(self, job_id: str, poll_every: int = POLL_EVERY) -> str:
        """
        Block until the job leaves the queue. Returns the final Slurm state string.
        """
        while True:
            try:
                state = self.ssh(f"squeue -j {job_id} -h -o '%T'")
            except RuntimeError:
                state = ""

            if not state:
                try:
                    final = self.ssh(
                        f"sacct -j {job_id} --noheader -o State | head -1"
                    ).split()[0]
                except RuntimeError:
                    final = "UNKNOWN"
                print(f"Job {job_id} finished  -  state: {final}")
                return final

            print(f"  Job {job_id}: {state}  -  polling in {poll_every}s")
            time.sleep(poll_every)

    def fetch_log(self, job_id: str, local_dir: str = "./results") -> Path:
        """
        Download the Slurm .out log for job_id to local_dir.
        Returns the local Path of the downloaded file.
        """
        scratch    = self.get_scratch()
        remote_out = self.ssh(f"find {scratch} -name '{job_id}.out' | head -1")
        if not remote_out:
            raise FileNotFoundError(f"No .out file found for job {job_id} under {scratch}")

        Path(local_dir).mkdir(parents=True, exist_ok=True)
        local_path = Path(local_dir) / f"{job_id}.out"

        with self.client.open_sftp() as sftp:
            sftp.get(remote_out, str(local_path))

        print(f"Fetched log: {local_path}")
        return local_path

    def download_directory(self, remote_dir: str, local_dir: str) -> list[str]:
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        downloaded: list[str] = []
        with self.client.open_sftp() as sftp:
            for entry in sftp.listdir_attr(remote_dir):
                if stat.S_ISDIR(entry.st_mode):
                    continue
                local_file = local_path / entry.filename
                sftp.get(f"{remote_dir}/{entry.filename}", str(local_file))
                downloaded.append(str(local_file))

        return downloaded

    def run_job(self, job: dict, local_script: str | None = None, local_dir: str = "./results") -> dict:
        """
        Full pipeline: upload (optional)  submit  wait  fetch log.
        Returns a result dict with job_id, state, and log_path.
        """
        job_id, _stderr = self.submit_job(job, local_script=local_script)
        state    = self.wait_for_job(job_id)
        log_path = self.fetch_log(job_id, local_dir=local_dir)

        return {
            "job_id":   job_id,
            "state":    state,
            "log_path": str(log_path),
            "success":  state == "COMPLETED",
        }