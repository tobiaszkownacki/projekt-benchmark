"""Runs the protocol validator against a submitted optimizer, in a sandbox.

The point of validating before queueing is not politeness to the user -- it is
that a broken submission which reaches SLURM burns grant hours that §5.2 says
are the scarce resource. Catching it here costs 30 seconds of a local CPU.

The code being checked is untrusted by construction: it is arbitrary Python
written by a competition entrant. It therefore runs under every restriction §7
lists -- no network, 2 GB, one CPU, read-only root, a writable tmpfs, a
non-root user and a hard timeout -- and its own source is mounted read-only.

The validator's output is a readable log, which is precisely what /submit needs
to show. It is returned verbatim rather than summarised.
"""

import asyncio
import hashlib
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.settings import find_source_root, settings

logger = logging.getLogger(__name__)

REPO_ROOT = find_source_root()
VALIDATOR_SCRIPT = (
    "src/benchmark_core/optimization_engine/optimizers/validation/verify_optimizer.py"
)

_FAMILY_HINTS = {
    "gradient": ("evaluate_with_grad", ".grad(", "grad()"),
    "gradient_free": ("evaluate(", "population", "sigma", "cma"),
}


@dataclass
class ValidationResult:
    ok: bool
    log: str
    output_type: Optional[str] = None
    family: Optional[str] = None
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

    A guess is acceptable here and would not be on the leaderboard: §13.1 makes
    method family the axis the whole thesis turns on, so it is stored as a
    column and is editable, not re-derived from source text at query time.
    """
    lowered = source.lower()
    if any(token in lowered for token in _FAMILY_HINTS["gradient"]):
        return "gradient"
    return "gradient_free"


def docker_available() -> bool:
    return shutil.which("docker") is not None


def _docker_command(workdir: Path, filename: str) -> list[str]:
    return [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", "2g",
        "--cpus", "1.0",
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--user", "65534:65534",
        "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL",
        "--pids-limit", "128",
        "-v", f"{REPO_ROOT}/src:/bench/src:ro",
        "-v", f"{workdir}:/submission:ro",
        "-w", "/bench",
        "-e", "PYTHONPATH=/bench/src",
        "-e", "HOME=/tmp",
        settings.validator_image,
        "python", "-c", _IN_CONTAINER_ENTRY, f"/submission/{filename}",
    ]


# Executed inside the container. Installs the import aliases first, because the
# validator and the evaluator it imports still use the pre-refactor module names.
_IN_CONTAINER_ENTRY = """
import runpy, sys
sys.path.insert(0, "/bench/src")
from compat.benchmark_aliases import install
install()
sys.argv = ["verify_optimizer", sys.argv[1]]
runpy.run_path(
    "/bench/" + "src/benchmark_core/optimization_engine/optimizers/validation/"
    "verify_optimizer.py",
    run_name="__main__",
)
"""


async def validate_source(source: str, filename: str = "optimizer.py") -> ValidationResult:
    """Validate uploaded source in the sandbox, or explain why we could not."""
    family = infer_family(source)

    if not settings.validator_enabled:
        return ValidationResult(
            ok=True,
            log="Walidator wyłączony konfiguracją (VALIDATOR_ENABLED=0).\n"
                "Zgłoszenie przyjęte bez kontroli protokołu.",
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

    workdir = Path(tempfile.mkdtemp(prefix="submission-"))
    try:
        target = workdir / filename
        target.write_text(source, encoding="utf-8")
        target.chmod(0o444)

        process = await asyncio.create_subprocess_exec(
            *_docker_command(workdir, filename),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=settings.validator_timeout + 10
            )
        except asyncio.TimeoutError:
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
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
