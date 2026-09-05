"""The lint gate must fail loudly when flake8 does not actually run.

flake8 exits 1 both when it finds violations and when it cannot start at all,
and in the second case stdout is empty -- indistinguishable from a clean tree
by exit code alone. The gate used to read only stdout, so an uninstalled or
crashing linter made CI pass having checked nothing. These tests pin the
distinction, because it is invisible in a green build.
"""

import importlib.util
import pathlib
import subprocess

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "lint_baseline.py"


def _load():
    spec = importlib.util.spec_from_file_location("lint_baseline", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lint_baseline = _load()


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(args=["flake8"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_clean_run_reports_no_violations(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(0))
    assert lint_baseline.run_flake8() == {}


def test_violations_are_counted_per_file_and_code(monkeypatch):
    output = (
        "./src/web/app/main.py:10:1: F401 unused\n"
        "./src/web/app/main.py:12:1: F401 unused\n"
        "./src/web/app/other.py:3:80: E501 too long\n"
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(1, output))
    assert lint_baseline.run_flake8() == {
        ("src/web/app/main.py", "F401"): 2,
        ("src/web/app/other.py", "E501"): 1,
    }


def test_uninstalled_linter_is_not_mistaken_for_a_clean_tree(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _completed(1, "", "No module named flake8"),
    )
    with pytest.raises(lint_baseline.Flake8Unusable, match="No module named flake8"):
        lint_baseline.run_flake8()


def test_a_crash_is_not_mistaken_for_a_clean_tree(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(2, "", "boom"))
    with pytest.raises(lint_baseline.Flake8Unusable, match="boom"):
        lint_baseline.run_flake8()


def test_the_gate_exits_non_zero_when_the_linter_is_unusable(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(1, "", "No module named flake8"))
    monkeypatch.setattr("sys.argv", ["lint_baseline.py"])
    assert lint_baseline.main() == 2
