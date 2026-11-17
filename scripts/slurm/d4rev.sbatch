#!/bin/bash -l
# D4rev (wB97M-D4rev / wB97X-D4rev): GPU tests incl. slow, then the molecular-batch benchmark.
# Submit from the repository root: sbatch scripts/slurm/d4rev.sbatch
#SBATCH --job-name=torch_dispersion_d4rev
#SBATCH --account=mpo_inst
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=daisg301,daisg302
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
module load gcc/15
export CXX="$(which g++)"
export OMP_NUM_THREADS=12
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run --extra torchsim pytest tests/test_d4_rev.py -m gpu -p no:logging -q -ra --durations=10
mkdir -p benchmarks/results
uv run --extra torchsim python scripts/benchmark_d4rev.py \
    --out "benchmarks/results/d4rev_${SLURM_JOB_ID}.json"
