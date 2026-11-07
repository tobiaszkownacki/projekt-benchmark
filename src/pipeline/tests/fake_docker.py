"""Stands in for the docker CLI, so the local executor can be tested without Docker.

It understands the calls the executor makes: `run` records its arguments and
starts the container's command as a plain detached process, with /job pointing
at the mounted host directory; `ps --all`, `inspect`, `logs` and `rm --force`
report and end those processes the way Docker reports and ends containers. The
sandbox itself is not reproduced; the tests check the arguments that ask Docker
for it, and a separate test runs the real image where it is built.
"""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

STATE = Path(os.environ["FAKE_DOCKER_STATE"])
CALLS = STATE / "run_calls.json"
FLAGS = {"--detach", "--read-only", "--rm", "--privileged"}


def _load(path: Path, empty):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else empty


def _container(name: str) -> Path:
    return STATE / f"container-{name}"


def _run(arguments: list[str]) -> int:
    calls = _load(CALLS, [])
    calls.append(arguments)
    CALLS.write_text(json.dumps(calls), encoding="utf-8")
    if os.environ.get("FAKE_DOCKER_REFUSE"):
        print("docker: Error response from daemon: refused by the test", file=sys.stderr)
        return 125

    options: dict[str, str] = {}
    index = 0
    while arguments[index].startswith("--"):
        if arguments[index] in FLAGS:
            index += 1
        else:
            options.setdefault(arguments[index], arguments[index + 1])
            index += 2
    command = arguments[index + 1 :]  # what follows the image
    host_dir, container_dir = options["--volume"].split(":")
    command = [options["--entrypoint"], *(host_dir if part == container_dir else part for part in command)]

    record = _container(options["--name"])
    record.mkdir()
    supervisor = subprocess.Popen(
        [sys.executable, __file__, "_supervise", str(record), *command],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    (record / "pid").write_text(str(supervisor.pid))
    print(f"id-{options['--name']}")
    return 0


def _supervise(record: Path, command: list[str]) -> int:
    with open(record / "log", "wb") as log:
        code = subprocess.call(command, stdout=log, stderr=log)
    # A process killed by signal N leaves 128 + N, as in a container.
    (record / "exit").write_text(str(code if code >= 0 else 128 - code))
    return 0


def _ps() -> int:
    for record in sorted(STATE.glob("container-*")):
        print(record.name.removeprefix("container-"), "exited" if (record / "exit").exists() else "running")
    return 0


def _inspect(name: str) -> int:
    record = _container(name)
    if not (record / "exit").exists():
        print(f"Error: No such object: {name}", file=sys.stderr)
        return 1
    print((record / "exit").read_text(), "false")
    return 0


def _logs(arguments: list[str]) -> int:
    record = _container(arguments[-1])
    if not record.exists():
        print(f"Error: No such container: {arguments[-1]}", file=sys.stderr)
        return 1
    lines = (record / "log").read_text(errors="replace").splitlines()
    print("\n".join(lines[-int(arguments[arguments.index("--tail") + 1]) :]))
    return 0


def _rm(name: str) -> int:
    record = _container(name)
    if not record.exists():
        print(f"Error response from daemon: No such container: {name}", file=sys.stderr)
        return 1
    session = int((record / "pid").read_text())
    # Docker kills everything in the container: here, everything in the session.
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            if int(stat.read_text().rsplit(")", 1)[1].split()[3]) == session:
                os.kill(int(stat.parent.name), signal.SIGKILL)
        except (OSError, ValueError):
            continue
    for path in record.iterdir():
        path.unlink()
    record.rmdir()
    return 0


def main(arguments: list[str]) -> int:
    command, rest = arguments[0], arguments[1:]
    if command == "_supervise":
        return _supervise(Path(rest[0]), rest[1:])
    if command == "run":
        return _run(rest)
    if command == "ps":
        return _ps()
    if command == "inspect":
        return _inspect(rest[-1])
    if command == "logs":
        return _logs(rest)
    if command == "rm":
        return _rm(rest[-1])
    print(f"fake docker does not know {command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
