"""Per-stage time and peak memory of the batched D4 model, to plan fused/Triton kernels.

Wraps the stage functions of ``d4_dispersion_batched`` with synchronising timers and reports,
per batch and s9, the time in: the neighbour list and its restrictions, the coincidence check,
the reference tables, the host-side EEQ setup and direct Coulomb sum, the compiled graph stage
(CN, EEQ, C6), the two-body and ATM tile loops and the chain closure (the backward of the graph
stage); the peak GPU memory of one call; and the number of CUDA kernel launches
(torch.profiler).

    uv run python scripts/profile_d4_stages.py --device cuda
"""

import argparse
import functools
import json
import statistics
import time
from collections import defaultdict
from typing import Any

import torch
from ase.build import bulk
from ase.io import read
from benchmark_torchsim import FIXTURES, build_batch

import torch_dispersion.d4.dispersion as d4_module
from torch_dispersion.d4 import d4_dispersion_batched
from torch_dispersion.neighbors import NeighborMatrix
from torch_dispersion.settings import D4Settings, NvalchemiNeighbors

TIMES: dict[str, float] = defaultdict(float)


def timed(name: str, fn, label=None):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        TIMES[label(args, kwargs) if label else name] += time.perf_counter() - t
        return out

    return wrapper


def instrument() -> None:
    d4 = d4_module
    d4.build_flat = timed("neighbour list", d4.build_flat)
    NeighborMatrix.restrict = timed("neighbour list restrict", NeighborMatrix.restrict)
    d4.check_coincident = timed("coincidence check", d4.check_coincident)
    d4._tables = timed("reference tables", d4._tables)
    labels = {"_graph": "graph stage forward (CN, EEQ, C6)"}
    graph_fn = d4.graph_fn

    def timed_graph_fn(fn, compile: bool):  # wrap the compiled function, not the traced one
        return timed(labels[fn.__name__], graph_fn(fn, compile))

    d4.graph_fn = timed_graph_fn  # ty: ignore[invalid-assignment]
    eeq = d4.eeq
    eeq.ewald_alphas = timed("EEQ setup: Ewald alphas", eeq.ewald_alphas)
    eeq.wigner_seitz_images = timed("EEQ setup: Wigner-Seitz pairs", eeq.wigner_seitz_images)
    eeq.direct_coulomb = timed("EEQ direct Coulomb forward", eeq.direct_coulomb)
    d4._two_body = timed("two-body tiles (whole batch)", d4._two_body)
    d4._three_body = timed("ATM tiles (whole batch)", d4._three_body)
    d4.close_chain = timed("chain closure", d4.close_chain)


def md_cells() -> list:
    """15 cells of 144-768 atoms (5460 atoms), the shape of li_diffusion's MD batches."""
    crystals = {a.info["name"]: a for a in read(FIXTURES / "structures.extxyz", ":")}
    reps = [
        ("benzamide", (3, 3, 1)), ("benzamide", (4, 4, 2)), ("benzamide", (4, 4, 3)),
        ("bromobenzene", (3, 3, 2)), ("bromobenzene", (4, 4, 3)), ("cyclohexane", (2, 2, 2)),
        ("cyclohexane", (3, 3, 2)), ("cyclohexane", (3, 3, 3)), ("dmso", (3, 3, 2)),
        ("dmso", (4, 4, 2)), ("dmso", (4, 4, 3)), ("emim_ntf2", (2, 2, 2)),
        ("emim_ntf2", (3, 2, 2)), ("glycerol", (3, 3, 2)), ("glycerol", (3, 3, 3)),
    ]  # fmt: skip
    return [crystals[f"{name}_crystal"].repeat(r) for name, r in reps]


def small_cells(n: int) -> list:
    """n periodic cells of 64 atoms: rock salt and silicon cubic 2x2x2 supercells."""
    cells = []
    for k in range(n):
        if k % 2:
            cells.append(bulk("NaCl", "rocksalt", a=5.64 + 0.01 * k, cubic=True).repeat(2))
        else:
            cells.append(bulk("Si", "diamond", a=5.43 + 0.01 * k, cubic=True).repeat((2, 2, 2)))
    return cells


def monomers(n: int) -> list:
    """n fixture monomers (10-19 atoms, neutral), molecules without PBC."""
    pool = [a for a in read(FIXTURES / "structures.extxyz", ":") if "_monomer" in a.info["name"]]
    pool = [a for a in pool if "emim" not in a.info["name"]]  # the ions (charged)
    out = []
    for k in range(n):
        atoms = pool[k % len(pool)].copy()
        atoms.pbc = False
        out.append(atoms)
    return out


def to_inputs(atoms_list):
    numbers = torch.cat([torch.tensor(a.numbers) for a in atoms_list])
    positions = torch.cat([torch.tensor(a.positions) for a in atoms_list])
    pbc = bool(atoms_list[0].pbc.all())
    cells = torch.stack([torch.tensor(a.cell.array) for a in atoms_list]) if pbc else None
    return numbers, positions, cells, pbc, [len(a) for a in atoms_list]


def profile(atoms_list, s9: float, repeats: int, functional: str) -> dict[str, Any]:
    settings = D4Settings.from_functional(
        functional, s9=s9, device="cuda", compile=True, neighbor_backend=NvalchemiNeighbors()
    )
    numbers, positions, cells, pbc, counts = to_inputs(atoms_list)

    def call():
        return d4_dispersion_batched(numbers, positions, cells, pbc, counts, settings)

    call()  # warm-up and compilation
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    call()
    torch.cuda.synchronize()
    peak_gib = (torch.cuda.max_memory_allocated() - base) / 2**30
    totals = []
    TIMES.clear()
    for _ in range(repeats):
        torch.cuda.synchronize()
        t = time.perf_counter()
        call()
        torch.cuda.synchronize()
        totals.append(time.perf_counter() - t)
    stages = {k: v / repeats for k, v in sorted(TIMES.items(), key=lambda kv: -kv[1])}
    stages["other (Python, restrict, pad, inputs)"] = statistics.median(totals) - sum(
        stages.values()
    )
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    launches = sum(e.count for e in prof.key_averages() if e.device_type.name == "CUDA")
    return {
        "systems": len(counts),
        "atoms": sum(counts),
        "s9": s9,
        "total_s": statistics.median(totals),
        "stages_s": stages,
        "cuda_kernel_launches": launches,
        "peak_gib": peak_gib,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default=None)
    parser.add_argument("--functional", default="r2scan")
    parser.add_argument("--batches", nargs="+", default=["md", "crystals", "small_cells"])
    args = parser.parse_args()
    instrument()
    batches = {
        "md": ("li_diffusion MD shape (15 cells, 144-768 atoms)", md_cells, (0.0,)),
        "crystals": ("fixture crystals (12 cells, 144-1000 atoms)", build_batch, (0.0, 1.0)),
        "small_cells": ("64 small cells (64 atoms)", lambda: small_cells(64), (0.0, 1.0)),
        "monomers": ("64 molecules (10-19 atoms, no PBC)", lambda: monomers(64), (0.0, 1.0)),
    }
    results = []
    for key in args.batches:
        name, make, s9_values = batches[key]
        atoms_list = make()
        for s9 in s9_values:
            r: dict[str, Any] = profile(atoms_list, s9, args.repeats, args.functional)
            r["batch"] = name
            results.append(r)
            print(
                f"\n== {name}, s9 = {s9}: {r['total_s']:.3f} s, {r['atoms']} atoms, "
                f"{r['cuda_kernel_launches']} CUDA kernel launches, "
                f"peak {r['peak_gib']:.2f} GiB above the inputs"
            )
            for k, v in r["stages_s"].items():
                print(f"  {k:40s} {v:8.4f} s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
