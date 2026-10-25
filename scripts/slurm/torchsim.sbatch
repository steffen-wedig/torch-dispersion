#!/bin/bash -l
# GPU tests of the batched TorchSim model (incl. slow) and its benchmark, on one H200/B200.
# Submit from the repository root: sbatch scripts/slurm/torchsim.sbatch
# Needs the torchsim extra (uv sync --extra torchsim) in $UV_PROJECT_ENVIRONMENT (default .venv).
#SBATCH --job-name=torch_dispersion_torchsim
#SBATCH --account=mpo_inst
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=daisg301,daisg302
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
module load gcc/15
export CXX="$(which g++)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run --extra torchsim pytest tests/test_torchsim.py tests/test_backends.py tests/test_d4_compile.py \
    -m gpu -p no:logging -q -ra --durations=15
mkdir -p benchmarks/results
uv run --extra torchsim python scripts/benchmark_torchsim.py --device cuda \
    --out "benchmarks/results/torchsim_${SLURM_JOB_ID}.json"
