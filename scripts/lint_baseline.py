#!/usr/bin/env python3
"""Run flake8 and fail only on violations that are not already known.

The repository carries pre-existing violations across the queue services, the
optimization engine and the Streamlit views. Excluding them would hide the debt
and failing on them would leave CI permanently red, so they are recorded in
.flake8-baseline and this script fails only when a file gains a violation it did
not have before. New code has to be clean; old code stays visible.

    python scripts/lint_baseline.py            check
    python scripts/lint_baseline.py --update   re-record after fixing something
"""

import argparse
import collections
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / ".flake8-baseline"
# Line numbers move whenever anything above them changes, so the baseline keys
# on (file, code) and counts occurrences instead.
LINE = re.compile(r"^(?P<file>[^:]+):\d+:\d+: (?P<code>[A-Z]+\d+) ")


class Flake8Unusable(RuntimeError):
    """flake8 did not run, so its silence means nothing."""


def run_flake8() -> collections.Counter:
    result = subprocess.run(
        [sys.executable, "-m", "flake8"],
        cwd=ROOT, capture_output=True, text=True,
    )
    found: collections.Counter = collections.Counter()
    for line in result.stdout.splitlines():
        match = LINE.match(line)
        if match:
            found[(match["file"].lstrip("./"), match["code"])] += 1

    # flake8 exits 1 both when it finds violations and when it cannot run at
    # all, so an uninstalled linter is indistinguishable from a clean tree by
    # exit code alone. A real run that exits non-zero always names at least one
    # violation, so the output is what decides.
    if result.returncode not in (0, 1) or (result.returncode == 1 and not found):
        raise Flake8Unusable(
            f"flake8 exited {result.returncode} without reporting a violation.\n"
            f"stderr: {result.stderr.strip() or '(empty)'}\n"
            f"stdout: {result.stdout.strip() or '(empty)'}"
        )
    return found


def read_baseline() -> collections.Counter:
    known: collections.Counter = collections.Counter()
    if not BASELINE.is_file():
        return known
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path, code, count = line.rsplit(" ", 2)
        known[(path, code)] = int(count)
    return known


def write_baseline(found: collections.Counter) -> None:
    lines = [
        "# Known flake8 violations, recorded so CI can fail on new ones without",
        "# hiding the existing debt. Regenerate with:",
        "#     python scripts/lint_baseline.py --update",
        "# Shrinking this file is always welcome; growing it needs a reason.",
    ]
    lines += [f"{path} {code} {count}" for (path, code), count in sorted(found.items())]
    BASELINE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    try:
        found = run_flake8()
    except Flake8Unusable as exc:
        print(f"Lint gate did not run: {exc}", file=sys.stderr)
        return 2

    if args.update:
        write_baseline(found)
        print(f"Recorded {sum(found.values())} violation(s) across {len(found)} file/code pairs.")
        return 0

    known = read_baseline()
    regressions = {
        key: (count, known.get(key, 0))
        for key, count in found.items()
        if count > known.get(key, 0)
    }

    if regressions:
        print("New flake8 violations:\n")
        for (path, code), (now, before) in sorted(regressions.items()):
            print(f"  {path}: {code} x{now} (baseline {before})")
        print("\nFix them, or run scripts/lint_baseline.py --update if the "
              "change is deliberate.")
        return 1

    fixed = sum(known.values()) - sum(found.values())
    print(f"No new violations. Known debt: {sum(found.values())} "
          f"across {len({p for p, _ in found})} file(s).")
    if fixed > 0:
        print(f"{fixed} fewer than the baseline -- run --update to lock that in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
