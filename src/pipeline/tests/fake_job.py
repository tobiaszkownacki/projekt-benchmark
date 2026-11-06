"""Stands in for the real job script, so the executor can be tested without torch.

Takes the same arguments the local executor passes and leaves the same three
things behind: artifacts, a result manifest and a terminal state.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.run_result import RunResult, RunSeries  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stop-condition", default="{}")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    (job_dir / "reports").mkdir(parents=True, exist_ok=True)
    (job_dir / "reports" / "loss.png").write_bytes(b"not really a plot")
    RunResult(
        stop_reason="EPOCH_LIMIT",
        gradient_count=12,
        database_reaches=34,
        final_loss=0.5,
        final_accuracy=0.75,
        total_steps=6,
        total_epochs=2,
        wall_time_seconds=1.5,
        series=RunSeries(
            epochs=[1, 2],
            loss=[0.9, 0.5],
            accuracy=[0.5, 0.75],
            gradient_count=[6, 12],
            database_reaches=[17, 34],
            wall_time_seconds=[0.7, 1.5],
        ),
    ).write_manifest(job_dir)
    (job_dir / "status.json").write_text(json.dumps({"state": "COMPLETED"}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
