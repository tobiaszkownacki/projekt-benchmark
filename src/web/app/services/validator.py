"""Runs the protocol validator against a submitted optimizer, in a sandbox.

A broken submission that reaches SLURM burns grant hours; catching it here
costs 30 seconds of local CPU.

The code being checked is untrusted by construction -- arbitrary Python written
by a competition entrant -- so it runs with no network, 2 GB, one CPU, a
read-only root with a writable tmpfs, a non-root user and a hard timeout, and
receives its source only through standard input.

The validator log is returned verbatim rather than summarised, because /submit
displays it to the participant.
"""

import asyncio
import hashlib
import logging
import shutil
from dataclasses import dataclass

from app.settings import settings

logger = logging.getLogger(__name__)

_FAMILY_HINTS = {
    "gradient": ("evaluate_with_grad", ".grad(", "grad()"),
    "gradient_free": ("evaluate(", "population", "sigma", "cma"),
}


@dataclass
class ValidationResult:
    ok: bool
    log: str
    output_type: str | None = None
    family: str | None = None
    version: str = "sandbox-1"

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "log": self.log,
            "output_type": self.output_type,
            "family": self.family,
            "version": self.version,
        }


def sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def infer_family(source: str) -> str:
    """Best-effort guess used only as a default the submitter can override.

    A guess is acceptable here and would not be on the leaderboard: method
    family is stored as a column and is editable, not re-derived from source
    text at query time.
    """
    lowered = source.lower()
    if any(token in lowered for token in _FAMILY_HINTS["gradient"]):
        return "gradient"
    return "gradient_free"


def docker_available() -> bool:
    return shutil.which("docker") is not None


def _docker_command() -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--interactive",
        "--network",
        "none",
        "--memory",
        "2g",
        "--cpus",
        "1.0",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--user",
        "65534:65534",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--pids-limit",
        "128",
        "-w",
        "/app",
        "-e",
        "PYTHONPATH=/app:/app/src",
        "-e",
        "HOME=/tmp",
        "--entrypoint",
        "python",
        settings.validator_image,
        "-c",
        _IN_CONTAINER_ENTRY,
    ]


# Executed inside the container.
_IN_CONTAINER_ENTRY = """
import pathlib, runpy, sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/src")
from compat.benchmark_aliases import install
install()
submission = pathlib.Path("/tmp/optimizer.py")
submission.write_bytes(sys.stdin.buffer.read())
submission.chmod(0o444)
sys.argv = ["verify_optimizer", str(submission)]
runpy.run_path(
    "/app/" + "src/benchmark_core/optimization_engine/optimizers/validation/"
    "verify_optimizer.py",
    run_name="__main__",
)
"""


async def validate_source(source: str) -> ValidationResult:
    """Validate uploaded source in the sandbox, or explain why we could not."""
    family = infer_family(source)

    if not settings.validator_enabled:
        return ValidationResult(
            ok=True,
            log="Walidator wyłączony konfiguracją (VALIDATOR_ENABLED=0).\nZgłoszenie przyjęte bez kontroli protokołu.",
            family=family,
            version="disabled",
        )

    if not docker_available():
        # Refusing here would block every submission on a host without Docker;
        # accepting silently would hide that nothing was checked. Say so.
        return ValidationResult(
            ok=True,
            log="Walidator niedostępny: brak polecenia `docker` na tym hoście.\n"
            "Zgłoszenie przyjęte BEZ kontroli protokołu — kod nie został "
            "sprawdzony.",
            family=family,
            version="unavailable",
        )

    process = await asyncio.create_subprocess_exec(
        *_docker_command(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(source.encode("utf-8")), timeout=settings.validator_timeout + 10
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return ValidationResult(
            ok=False,
            log=f"Walidacja przerwana po {settings.validator_timeout} s.\n"
            "Optymalizator nie zakończył pojedynczego kroku w limicie czasu.",
            family=family,
        )

    log = stdout.decode("utf-8", errors="replace")
    ok = process.returncode == 0 and "ERROR" not in log
    return ValidationResult(ok=ok, log=log, family=family)
