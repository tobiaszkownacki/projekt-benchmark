import subprocess
import tempfile
from pathlib import Path

def run_synchronous_validation(user_code: str, optimizer_filename: str = "custom_opt.py") -> dict:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / optimizer_filename

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(user_code)

        cmd = [
            "docker", "run",
            "--rm",
            "--network", "none",
            "--memory", "2g",
            "--cpus", "1.0",
            "--read-only",
            "--tmpfs", "/tmp:mode=1777",
            "-e", "CUPY_CACHE_DIR=/tmp",
            "-e", "MPLCONFIGDIR=/tmp",
            "-v", f"{file_path}:/app/{optimizer_filename}:ro",
            "optimizer-validator",
            optimizer_filename
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30.0
            )

            if result.returncode == 0:
                return {"status": "success", "logs": result.stdout}
            else:
                return {"status": "error", "logs": result.stdout + result.stderr}

        except subprocess.TimeoutExpired:
            return {"status": "timeout", "logs": "Error: Time limit exceeded (30 seconds). Does your code contain an infinite loop??"}
