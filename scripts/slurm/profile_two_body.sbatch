#!/bin/bash -l
# Two-body tile loop trace of the batched D4 model (scripts/profile_d4_two_body.py) on gpudev.
# Submit from the repository root: sbatch scripts/slurm/profile_two_body.sbatch
#SBATCH --job-name=torch_dispersion_profile_two_body
#SBATCH --account=mpo_inst
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=daisg301,daisg302
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
module load gcc/15
export CXX="$(which g++)"
cd scripts
uv run --extra torchsim python profile_d4_two_body.py
