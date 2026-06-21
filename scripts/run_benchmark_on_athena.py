# run_benchmark.py
from scripts.athena import run_job
import os
from dotenv import load_dotenv

load_dotenv()
ATHENA_PATH = os.getenv("ATHENA_REMOTE_PATH")

job = {
    "job_name": "benchmark_run",
    "account":  "plggolemml26-gpu-a100",
    "time":     "02:00:00",
    "mem":      "32G",
    "cpus":     8,
    "gpus":     1,

    "workdir":  f"{ATHENA_PATH}/projekt-benchmark",

    "pre_commands": [
        "uv sync"
    ],

    "run_command": "uv run -m src.benchmark.run_benchmark --dataset wine_quality --optimizer adam sgd cma-es --max-epochs 10 --max-gradients 100000 --plot"
}

if __name__ == "__main__":
    print("Submitting projekt-benchmark to Athena...")
    result = run_job(job)

    print("\n=== JOB FINISHED ===")
    print(f"Status: {result['state']}")
    print(f"Log saved at: {result['log_path']}")