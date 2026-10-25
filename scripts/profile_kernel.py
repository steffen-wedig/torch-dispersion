"""Profile one liquid-box evaluation (GPU) and print the top CUDA kernels.

uv run python scripts/profile_kernel.py --grad --block 16 --tile 256
"""

import argparse
from pathlib import Path

import torch
from ase.io import read
from torch.profiler import ProfilerActivity, profile

from torch_dispersion.atm import atm_dispersion
from torch_dispersion.settings import ATMSettings

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="dmso_liquid")
    p.add_argument("--cutoff", type=float, default=40.0)
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--dtype", default="float64")
    p.add_argument("--grad", action="store_true")
    p.add_argument("--eager", action="store_true")
    args = p.parse_args()
    atoms = next(
        a for a in read(FIXTURES / "structures.extxyz", ":") if a.info["name"] == args.name
    )
    settings = ATMSettings(
        three_body_cutoff_bohr=args.cutoff,
        centre_block_size=args.block,
        neighbor_tile_size=args.tile,
        dtype=args.dtype,
        compile=not args.eager,
    )

    def call():
        return atm_dispersion(
            torch.tensor(atoms.numbers),
            torch.tensor(atoms.positions),
            torch.tensor(atoms.cell.array),
            True,
            settings,
            compute_forces=args.grad,
            compute_stress=args.grad,
        )

    call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    print(
        prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=90)
    )


if __name__ == "__main__":
    main()
