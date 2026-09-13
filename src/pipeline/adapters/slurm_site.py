from dataclasses import dataclass


@dataclass(frozen=True)
class SlurmCredentials:
    """Names of the environment variables holding one centre's SSH credentials.

    Names rather than values: PLGrid-style credentials are personal, so they
    never reach the repository, and two centres configured in the same process
    keep their own.
    """

    host: str
    user: str
    password: str
    account: str
    remote_path: str


@dataclass(frozen=True)
class SlurmSite:
    """Everything that differs between one computing centre and the next.

    The mechanism -- sbatch, sacct, SFTP over SSH -- is the same at every SLURM
    site, so a new supercomputer is an instance of this class and a registration
    entry, not a subclass of the executor.
    """

    name: str
    partition: str
    credentials: SlurmCredentials
    project_dir_name: str = "projekt-benchmark"
    # Scratch is resolved on the login node because its path is per-user and
    # only the centre's own profile knows how to name it.
    scratch_command: str = "echo $SCRATCH"
    pre_commands: tuple[str, ...] = ()
    time_limit: str = "00:30:00"
    cpus: int = 1
    gpus: int = 1
    memory: str = "64G"
