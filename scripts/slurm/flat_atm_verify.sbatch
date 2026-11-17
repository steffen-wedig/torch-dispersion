#!/bin/bash -l
# D4 verification on a compute node: the D4rev and TorchSim ATM benchmarks on a B200 (the
# numbers in docs/d4.md are from job 913640), then the CPU suite and the GPU suite (incl. slow).
# Submit from the repository root: sbatch scripts/slurm/flat_atm_verify.sbatch
# (or SNAP=<worktree> sbatch --export=ALL ... to test a snapshot of the tree).
#SBATCH --job-name=torch_dispersion_flat_atm_verify
#SBATCH --account=mpo_inst
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --exclude=daisg301,daisg302
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
SNAP=${SNAP:-$PWD}
module load gcc/15
export CXX="$(which g++)"
export OMP_NUM_THREADS=12
nvidia-smi --query-gpu=name --format=csv,noheader
RUN="env PYTHONPATH=$SNAP/src uv run --extra torchsim"
PYTEST="pytest -c $SNAP/pyproject.toml --rootdir $SNAP $SNAP/tests -p no:logging -p no:cacheprovider -q -ra"
$RUN python -c "import torch_dispersion; print(torch_dispersion.__file__)"
echo "=== benchmarks, after"
$RUN python $SNAP/scripts/benchmark_d4rev.py --functional wb97m-rev \
    --out "benchmarks/results/d4rev_after_${SLURM_JOB_ID}.json"
$RUN python $SNAP/scripts/benchmark_torchsim.py --device cuda --s9 1 \
    --out "benchmarks/results/torchsim_atm_after_${SLURM_JOB_ID}.json"
echo "=== CPU suite"
$RUN $PYTEST -m "not gpu and not slow"
echo "=== GPU suite"
$RUN $PYTEST -m gpu --durations=10
