#!/bin/bash -l
# Benchmark on one H200 (pinned: timings depend on the hardware).
# Submit from the repository root: sbatch scripts/slurm/benchmark_d4.sbatch [benchmark_d4.py args]
#SBATCH --job-name=d4_benchmark
#SBATCH --account=mpo_inst
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=daisg301,daisg302
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
module load gcc/15
export CXX="$(which g++)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run python scripts/benchmark_d4.py --out "benchmarks/results/d4_h200_${SLURM_JOB_ID}.json" "$@"
