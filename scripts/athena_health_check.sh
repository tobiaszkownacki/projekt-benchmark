#!/bin/bash
#SBATCH --job-name=health_check
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=%j.out

uv run -m src.benchmark.run_benchmark --dataset digits --optimizer adam --task-id $(date +%s%N) --plot
