import os
import re
import stat
from pathlib import Path

import paramiko

from pipeline.adapters.slurm_site import SlurmSite
from shared.connectors.base import BaseConnector


class SlurmConnector(BaseConnector):
    """One SSH session to a SLURM login node.

    Credentials are read from the environment variables the site names, so the
    same class serves every centre and none of them is written down here.
    """

    def __init__(self, site: SlurmSite) -> None:
        self.site = site
        self.host = os.environ.get(site.credentials.host)
        self.user = os.environ.get(site.credentials.user)
        self.password = os.environ.get(site.credentials.password)
        self.account = os.environ.get(site.credentials.account, "")
        self.remote_path = os.environ.get(site.credentials.remote_path, "")
        self.client = None
        self._scratch: str | None = None

    @property
    def project_dir(self) -> str:
        return f"{self.remote_path}/{self.site.project_dir_name}"

    def __enter__(self) -> "SlurmConnector":
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
        """Run a command on the login node, return stdout. Raises on non-zero exit."""
        _, stdout, stderr = self.client.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode().strip()
        if exit_code != 0:
            raise RuntimeError(stderr.read().decode().strip())
        return out

    def ssh_capture(self, cmd: str) -> tuple[str, str, int]:
        """Run a command, returning (stdout, stderr, exit_code) without raising."""
        _, stdout, stderr = self.client.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        return stdout.read().decode().strip(), stderr.read().decode().strip(), exit_code

    def get_scratch(self) -> str:
        if self._scratch is None:
            self._scratch = self.ssh(self.site.scratch_command)
        return self._scratch

    def job_script(self, job_name: str, run_command: str, env_vars: dict[str, str]) -> str:
        site = self.site
        directives = [
            f"--job-name={job_name}",
            f"--partition={site.partition}",
            "--nodes=1",
            f"--cpus-per-task={site.cpus}",
            f"--mem={site.memory}",
            f"--time={site.time_limit}",
            f"--output={self.project_dir}/reports/{job_name}/%j.out",
        ]
        if self.account:
            directives.append(f"--account={self.account}")
        if site.gpus:
            directives.append(f"--gres=gpu:{site.gpus}")

        body = "\n".join(
            [
                "#!/bin/bash",
                *(f"#SBATCH {directive}" for directive in directives),
                "",
                *(f'export {name}="{value}"' for name, value in env_vars.items()),
                "",
                f"cd {self.project_dir}",
                "",
                *site.pre_commands,
                "",
                run_command,
            ]
        )
        return body.strip()

    def submit_job(self, job_name: str, run_command: str, env_vars: dict[str, str]) -> tuple[str, str]:
        """Submit one job. Returns (executor_task_id, stderr).

        stderr is returned rather than discarded because sbatch writes warnings
        there on submissions that succeed.
        """
        script = self.job_script(job_name, run_command, env_vars)
        job_dir = f"{self.get_scratch()}/{job_name}"

        # The output directory has to exist before sbatch starts: Slurm does not
        # create the one named by --output and the job dies without a log.
        self.ssh(f"mkdir -p {self.project_dir}/reports/{job_name}")

        escaped = script.replace("'", "'\\''")
        self.ssh(f"mkdir -p {job_dir} && printf '%s' '{escaped}' > {job_dir}/job.sh")

        stdout, stderr, exit_code = self.ssh_capture(f"cd {job_dir} && sbatch job.sh")
        if exit_code != 0:
            raise RuntimeError(stderr or f"sbatch exited with status {exit_code}")

        match = re.search(r"(\d+)", stdout)
        if not match:
            raise RuntimeError(f"Could not parse job ID from sbatch output: {stdout!r}")
        return match.group(1), stderr

    def download_results(self, remote_dir: str, local_dir: str) -> list[str]:
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
