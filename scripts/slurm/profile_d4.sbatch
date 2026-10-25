#!/bin/bash -l
# Per-stage profile of the batched D4 model (scripts/profile_d4_stages.py) on one H200/B200.
# Submit from the repository root: sbatch scripts/slurm/profile_d4.sbatch
#SBATCH --job-name=torch_dispersion_profile_d4
#SBATCH --account=mpo_inst
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=daisg301,daisg302
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
module load gcc/15
export CXX="$(which g++)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
mkdir -p benchmarks/results
cd scripts
uv run --extra torchsim python profile_d4_stages.py --out "../benchmarks/results/profile_d4_${SLURM_JOB_ID}.json"
