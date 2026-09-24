import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.services import validator


def test_docker_command_runs_prebuilt_image_and_reads_stdin(monkeypatch):
    monkeypatch.setattr(
        validator,
        "settings",
        SimpleNamespace(validator_image="test-validator:latest"),
    )

    command = validator._docker_command()

    assert command[:4] == ["docker", "run", "--rm", "--interactive"]
    assert "test-validator:latest" in command
    assert command[command.index("--entrypoint") + 1] == "python"
    assert "-v" not in command
    assert "sys.stdin.buffer.read()" in validator._IN_CONTAINER_ENTRY


def test_container_entry_imports_only_modules_the_image_ships():
    shipped = {p.name for p in (Path(__file__).resolve().parents[2]).iterdir() if p.is_dir()}
    imported = set()
    for node in ast.walk(ast.parse(validator._IN_CONTAINER_ENTRY)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])

    missing = imported - sys.stdlib_module_names - shipped
    assert not missing, f"the validator image has no module {sorted(missing)}"


@pytest.mark.asyncio
async def test_validate_source_sends_source_to_container_stdin(monkeypatch):
    source = "class Optimizer:\n    pass\n"
    received: dict[str, object] = {}

    class Process:
        returncode = 0

        async def communicate(self, data):
            received["stdin"] = data
            return b"PASSED test\n", None

    async def create_subprocess_exec(*command, **kwargs):
        received["command"] = command
        received["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(
        validator,
        "settings",
        SimpleNamespace(
            validator_enabled=True,
            validator_image="test-validator:latest",
            validator_timeout=30,
        ),
    )
    monkeypatch.setattr(validator, "docker_available", lambda: True)
    monkeypatch.setattr(validator.asyncio, "create_subprocess_exec", create_subprocess_exec)

    result = await validator.validate_source(source)

    assert result.ok is True
    assert result.log == "PASSED test\n"
    assert received["stdin"] == source.encode("utf-8")
    assert "test-validator:latest" in received["command"]
    assert received["kwargs"]["stdin"] is validator.asyncio.subprocess.PIPE
