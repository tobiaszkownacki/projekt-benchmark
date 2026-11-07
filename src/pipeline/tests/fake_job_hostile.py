"""Jobs that try to mislead or harm the executor. Which one runs is chosen by FAKE_JOB_MODE.

- liar: claims success in the files it controls, then fails.
- stubborn: ignores SIGTERM and never ends on its own.
- flood: writes more than the output limit, then keeps running.
- fifo: leaves a named pipe among its output, which nothing will ever write to.
- links: plants symlinks to a host file where the executor used to read and
  write, then fails, so its log tail is read.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    args, _ = parser.parse_known_args()
    job_dir = Path(args.job_dir)
    mode = os.environ["FAKE_JOB_MODE"]

    if mode == "liar":
        (job_dir / "status.json").write_text(json.dumps({"state": "COMPLETED"}))
        sys.exit(1)
    if mode == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(0.2)
    if mode == "flood":
        (job_dir / "big.bin").write_bytes(b"x" * (2 * 1024 * 1024))
        time.sleep(30)
    if mode == "fifo":
        os.mkfifo(job_dir / "pipe")
        (job_dir / "after.txt").write_text("fetched")
        sys.exit(0)
    if mode == "links":
        target = os.environ["FAKE_JOB_HOST_FILE"]
        for name in ("run.out", "status.json", "job.json", "leak"):
            (job_dir / name).symlink_to(target)
        (job_dir / "subdir").symlink_to(Path(target).parent, target_is_directory=True)
        print("a normal log line")
        sys.exit(2)
