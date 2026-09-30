"""Benchmark torch-atm on the fixture liquid boxes (design §8, "Performance").

Wall time and peak GPU memory per box for cutoff3 in {25, 30, 40} bohr, float64/float32,
eager/compiled, energy-only vs energy + forces + stress; triangles per second; and the CPU oracle
time for comparison. Writes a JSON file and prints a table.

    uv run python scripts/benchmark.py --out benchmarks/results/h200.json
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ase.io import read

import torch_atm.atm as atm_module
from torch_atm.atm import atm_dispersion
from torch_atm.kernel import TileInputs
from torch_atm.settings import ATMSettings

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
BOHR, HARTREE = 0.52917721, 27.211384500  # the fixtures' (FHI-aims) constants


def count_triangles(atoms, settings: ATMSettings) -> int:
    """Number of triangles inside the cutoff (each counted once), via a counting step."""

    def counting_step(acc, v_n, v_m, c6, t: TileInputs, s):
        cutoff2 = s.cutoff * s.cutoff
        eps = torch.finfo(v_n.dtype).eps
        r2_ij, r2_ik = v_n.square().sum(-1), v_m.square().sum(-1)
        r2_jk = (v_m[:, None] - v_n[:, :, None]).square().sum(-1)
        ok_ij = t.mask_n & (r2_ij <= cutoff2) & (r2_ij > eps)
        ok_ik = t.mask_m & (r2_ik <= cutoff2) & (r2_ik > eps)
        order = torch.arange(v_n.shape[1], device=v_n.device)
        upper = order[:, None] < order[None, :]
        valid = ok_ij[:, :, None] & ok_ik[:, None, :] & (r2_jk <= cutoff2) & (r2_jk > eps)
        valid = valid & (upper | ~t.diagonal)
        return acc + valid.sum(dtype=torch.float64)

    def counting_step_fn(with_grad: bool, compile: bool, gradient: str = "analytic"):
        return counting_step

    with patch.object(atm_module, "get_step_fn", counting_step_fn):
        result = run(atoms, settings.model_copy(update={"compile": False}), grad=False)
    # the energy slot now holds the count (times hartree_in_ev); each triangle has 3 vertices
    return round(float(result.energy) / settings.hartree_in_ev)


def run(atoms, settings, grad: bool):
    return atm_dispersion(
        torch.tensor(atoms.numbers),
        torch.tensor(atoms.positions),
        torch.tensor(atoms.cell.array),
        True,
        settings,
        compute_forces=grad,
        compute_stress=grad,
    )


def timed(fn, repeats: int) -> tuple[float, float]:
    """(median seconds, peak MiB) after one warm-up call."""
    cuda = torch.cuda.is_available()
    fn()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else float("nan")
    return statistics.median(times), peak


def oracle_time(atoms, disp3: float) -> float:
    from dftd3.interface import DispersionModel, RationalDampingParam

    b = BOHR
    model = DispersionModel(
        atoms.numbers, atoms.positions / b, atoms.cell.array / b, np.array([True] * 3)
    )
    model.set_realspace_cutoff(60.0, disp3, 40.0)
    param = RationalDampingParam(
        s6=1.0, s8=0.78981345, s9=1.0, a1=0.49484001, a2=5.73083694, alp=14.0
    )
    t0 = time.perf_counter()
    model.get_dispersion(param, grad=True)
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--liquids", nargs="*", default=None, help="subset of liquid names")
    parser.add_argument("--cutoffs", type=float, nargs="*", default=[25.0, 30.0, 40.0])
    parser.add_argument("--dtypes", nargs="*", default=["float64", "float32"])
    parser.add_argument("--compile", nargs="*", default=["eager", "compiled"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--oracle", action="store_true", help="also time the CPU oracle")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    atoms_all = [a for a in read(FIXTURES / "structures.extxyz", ":") if "liquid" in a.info["name"]]
    if args.liquids:
        atoms_all = [a for a in atoms_all if a.info["name"] in args.liquids]

    rows = []
    for atoms in atoms_all:
        name = atoms.info["name"]
        for cutoff in args.cutoffs:
            base = ATMSettings(
                device=device,
                three_body_cutoff_bohr=cutoff,
                centre_block_size=args.block,
                neighbor_tile_size=args.tile,
                bohr_in_angstrom=BOHR,
                hartree_in_ev=HARTREE,
            )
            n_tri = count_triangles(atoms, base)
            oracle_s = oracle_time(atoms, cutoff) if args.oracle else None
            for dtype in args.dtypes:
                for mode in args.compile:
                    settings = base.model_copy(
                        update={"dtype": dtype, "compile": mode == "compiled"}
                    )
                    for grad in (False, True):
                        seconds, peak = timed(
                            lambda a=atoms, s=settings, g=grad: run(a, s, g), args.repeats
                        )
                        row = {
                            "name": name,
                            "atoms": len(atoms),
                            "cutoff3_bohr": cutoff,
                            "dtype": dtype,
                            "mode": mode,
                            "forces_stress": grad,
                            "seconds": seconds,
                            "peak_mib": peak,
                            "triangles": n_tri,
                            "triangles_per_s": n_tri / seconds,
                            "oracle_seconds": oracle_s,
                        }
                        rows.append(row)
                        print(
                            f"{name:22s} rc={cutoff:4.0f} {dtype:7s} {mode:8s} "
                            f"{'E+F+S' if grad else 'E':5s} {seconds:8.3f}s {peak:9.0f} MiB "
                            f"{n_tri / seconds:10.3e} tri/s"
                            + (f"  oracle {oracle_s:.1f}s" if oracle_s else ""),
                            flush=True,
                        )
    meta = {
        "device": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "block": args.block,
        "tile": args.tile,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
