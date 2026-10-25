#!/bin/bash -l
# GPU test suite (pytest -m gpu, including slow GPU tests) on one H200/B200.
# Submit from the repository root: sbatch scripts/slurm/gpu_tests.sbatch [extra pytest args]
#SBATCH --job-name=torch_dispersion_gpu_tests
#SBATCH --account=mpo_inst
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=daisg301,daisg302
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
module load gcc/15
export CXX="$(which g++)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run pytest -m gpu -p no:logging -q -ra --durations=30 "$@"
