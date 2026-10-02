"""A job that is still running when the test cancels it, and says so if it ever finishes."""

import argparse
import time
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    args, _ = parser.parse_known_args()
    time.sleep(2)
    (Path(args.job_dir) / "finished").write_text("not cancelled", encoding="utf-8")
