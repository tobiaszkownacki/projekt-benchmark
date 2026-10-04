"""Run one benchmark task on this machine and leave what a downloader expects.

The local executor's job script, the counterpart of what sbatch starts on a
cluster. It writes into a staging directory that stands in for the cluster's
reports tree, so the download phase has something real to fetch, and it records
its own terminal state there because the poller has no other way to learn that
the run is over.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from shared.run_result import RunResult, RunSeries  # noqa: E402
from tools.local_backend.artifacts import write_run_artifacts  # noqa: E402
from tools.local_backend.runner import LocalBenchmarkRunner, StopCondition  # noqa: E402

STATUS_FILE = "status.json"


def _write_state(job_dir: Path, state: str) -> None:
    (job_dir / STATUS_FILE).write_text(json.dumps({"state": state}), encoding="utf-8")


def _to_run_result(result) -> RunResult:
    return RunResult(
        stop_reason=result.stop_reason,
        gradient_count=result.gradient_count,
        database_reaches=result.database_reaches,
        final_loss=result.final_loss,
        final_accuracy=result.final_accuracy,
        total_steps=result.total_steps,
        total_epochs=result.total_epochs,
        wall_time_seconds=result.wall_time_seconds,
        series=RunSeries(
            epochs=result.epoch_history,
            loss=result.loss_history,
            accuracy=result.accuracy_history,
            gradient_count=result.gradient_history,
            database_reaches=result.database_reaches_history,
            wall_time_seconds=result.time_history,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one local benchmark job")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stop-condition", default="{}")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        budget = json.loads(args.stop_condition)
        runner = LocalBenchmarkRunner(
            dataset_name=args.dataset,
            model_name=args.model,
            stop_condition=StopCondition(**budget),
            seed=args.seed,
        )
        result = runner.run(args.optimizer)
        write_run_artifacts(job_dir, result)
        _to_run_result(result).write_manifest(job_dir)
    except Exception:
        (job_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        _write_state(job_dir, "FAILED")
        return 1

    _write_state(job_dir, "COMPLETED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
