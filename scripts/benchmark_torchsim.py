"""Benchmark the batched TorchSim D4 model against a loop of single-system calls.

The batch is a set of periodic cells of ~100-1000 atoms built by replicating the fixture
crystals, the size of an in-flight autobatcher batch. Both paths compute energy, forces and
stress in float64 with the same settings:

- *loop*: ``d4_dispersion`` once per system (what a consumer-side wrapper does);
- *batched*: ``D4Model(settings)(state)`` (one batched neighbour list, per-system stages).

    uv run python scripts/benchmark_torchsim.py --device cuda --out benchmarks/results/ts.json
    uv run python scripts/benchmark_torchsim.py --device cpu --cutoffs short --s9 0
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch_sim as ts
from ase.io import read

from torch_dispersion.calculators.torchsim import D4Model
from torch_dispersion.d4 import d4_dispersion
from torch_dispersion.settings import BruteForceNeighbors, D4Settings, NvalchemiNeighbors

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
# r2SCAN-D4
PARAMS = {"s8": 0.60187490, "a1": 0.51559235, "a2": 5.77342911}
CUTOFFS = {
    "default": {},  # 60 / 40 / 30 bohr
    "short": {
        "two_body_cutoff_bohr": 25.0,
        "three_body_cutoff_bohr": 15.0,
        "coordination_cutoff_bohr": 18.0,
    },
}
# (fixture crystal, replication) -> 12 cells of 144-1000 atoms, 6090 atoms in all
BATCH = [
    ("dmso_crystal", (3, 3, 2)),
    ("dmso_crystal", (5, 5, 4)),
    ("benzamide_crystal", (2, 3, 2)),
    ("benzamide_crystal", (4, 4, 3)),
    ("bromobenzene_crystal", (2, 2, 3)),
    ("bromobenzene_crystal", (4, 4, 4)),
    ("cyclohexane_crystal", (2, 2, 2)),
    ("cyclohexane_crystal", (4, 4, 3)),
    ("emim_ntf2_crystal", (2, 2, 2)),
    ("emim_ntf2_crystal", (3, 3, 3)),
    ("glycerol_crystal", (2, 2, 3)),
    ("glycerol_crystal", (4, 4, 3)),
]


def build_batch():
    crystals = {a.info["name"]: a for a in read(FIXTURES / "structures.extxyz", ":")}
    return [crystals[name].repeat(rep) for name, rep in BATCH]


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, repeats: int, device: torch.device) -> tuple[float, float, float | None]:
    """(median, min seconds, peak MiB on CUDA) of ``fn()`` after one warm-up call."""
    fn()
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        sync(device)
        t = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t)
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    return statistics.median(times), min(times), peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cutoffs", choices=list(CUTOFFS), default="default")
    parser.add_argument("--s9", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--backend", choices=["nvalchemi", "brute_force"], default="nvalchemi")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--threads", type=int, help="torch CPU threads (default: torch's)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    atoms_list = build_batch()
    sizes = [len(a) for a in atoms_list]
    print(f"{len(sizes)} cells, {sum(sizes)} atoms: {sizes}")
    state = ts.io.atoms_to_state(atoms_list, device, torch.float64)
    inputs = [
        (torch.tensor(a.numbers), torch.tensor(a.positions), torch.tensor(a.cell.array))
        for a in atoms_list
    ]
    backend = NvalchemiNeighbors() if args.backend == "nvalchemi" else BruteForceNeighbors()
    rows = []
    for s9 in args.s9:
        settings = D4Settings.model_validate(
            {
                **PARAMS,
                **CUTOFFS[args.cutoffs],
                "s9": s9,
                "device": args.device,
                "compile": not args.no_compile,
                "neighbor_backend": backend,
            }
        )
        model = D4Model(settings)

        def loop(settings=settings):
            return [d4_dispersion(z, r, c, True, settings) for z, r, c in inputs]

        def batched(model=model):
            return model(state)

        ref, out = loop(), batched()
        energy = torch.stack([r.energy for r in ref])
        forces = torch.cat([r.forces for r in ref])
        stress = torch.stack([r.stress for r in ref])
        diff = {
            "energy": float((out["energy"] - energy).abs().max()),
            "forces": float((out["forces"] - forces).abs().max()),
            "stress": float((out["stress"] - stress).abs().max()),
        }
        t_loop, min_loop, peak_loop = timed(loop, args.repeats, device)
        t_batch, min_batch, peak_batch = timed(batched, args.repeats, device)
        row = {
            "s9": s9,
            "loop_s": t_loop,
            "loop_min_s": min_loop,
            "batched_s": t_batch,
            "batched_min_s": min_batch,
            "speedup": t_loop / t_batch,
            "loop_peak_mib": peak_loop,
            "batched_peak_mib": peak_batch,
            "max_abs_diff": diff,
        }
        rows.append(row)
        print(json.dumps(row))
    meta = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "host": platform.node(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "cutoffs": args.cutoffs,
        "backend": args.backend,
        "compile": not args.no_compile,
        "sizes": sizes,
        "n_atoms": int(np.sum(sizes)),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
