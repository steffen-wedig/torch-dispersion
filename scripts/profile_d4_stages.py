"""Per-stage time of the batched D4 model, to plan fused/Triton kernels.

Wraps the stage functions of ``d4_dispersion_batched`` with synchronising timers and reports,
per batch and s9, the time in: the neighbour list, the coincidence check, the graph stage
(split into CN, EEQ charges and C6), the two-body and ATM tile loops, the chain closure; plus the
number of CUDA kernel launches of one batched call (torch.profiler).

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
from benchmark_torchsim import build_batch

import torch_dispersion.d4.batched as batched_module
import torch_dispersion.d4.dispersion as d4_module
from torch_dispersion.d4 import d4_dispersion_batched
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
    batched_module.build_batched = timed("neighbour list", batched_module.build_batched)
    d4.check_coincident = timed("coincidence check", d4.check_coincident)
    d4._pad_system = timed("padding + reference C6", d4._pad_system)
    labels = {"_coordination_numbers": "CN (D4 + EEQ)", "_c6_pair": "C6(q), C6(0)"}
    graph_fn = d4.graph_fn

    def timed_graph_fn(fn, compile: bool):  # wrap the compiled function, not the traced one
        return timed(labels[fn.__name__], graph_fn(fn, compile))

    d4.graph_fn = timed_graph_fn  # ty: ignore[invalid-assignment]
    d4._charges = timed("EEQ charges", d4._charges)
    run_tiles = timed(
        "tiles", d4.run_tiles, lambda a, k: "two-body tiles" if a[5] == 1 else "ATM tiles"
    )
    d4.run_tiles = run_tiles
    d4.TileLayout.build = timed("tile layout (reshape)", d4.TileLayout.build)
    d4.close_chain = timed("chain closure", d4.close_chain)


def small_cells(n: int) -> list:
    """n periodic cells of 64 atoms: rock salt and silicon cubic 2x2x2 supercells."""
    cells = []
    for k in range(n):
        if k % 2:
            cells.append(bulk("NaCl", "rocksalt", a=5.64 + 0.01 * k, cubic=True).repeat(2))
        else:
            cells.append(bulk("Si", "diamond", a=5.43 + 0.01 * k, cubic=True).repeat((2, 2, 2)))
    return cells


def to_inputs(atoms_list):
    numbers = torch.cat([torch.tensor(a.numbers) for a in atoms_list])
    positions = torch.cat([torch.tensor(a.positions) for a in atoms_list])
    cells = torch.stack([torch.tensor(a.cell.array) for a in atoms_list])
    return numbers, positions, cells, [len(a) for a in atoms_list]


def profile(atoms_list, s9: float, repeats: int) -> dict[str, Any]:
    settings = D4Settings.from_functional(
        "r2scan", s9=s9, device="cuda", compile=True, neighbor_backend=NvalchemiNeighbors()
    )
    numbers, positions, cells, counts = to_inputs(atoms_list)

    def call():
        return d4_dispersion_batched(numbers, positions, cells, True, counts, settings)

    call()  # warm-up and compilation
    torch.cuda.synchronize()
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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    instrument()
    batches = {
        "fixture crystals (12 cells, 144-1000 atoms)": build_batch(),
        "64 small cells (64 atoms)": small_cells(64),
    }
    results = []
    for name, atoms_list in batches.items():
        for s9 in (0.0, 1.0):
            r: dict[str, Any] = profile(atoms_list, s9, args.repeats) | {"batch": name}
            results.append(r)
            print(
                f"\n== {name}, s9 = {s9}: {r['total_s']:.3f} s, {r['atoms']} atoms, "
                f"{r['cuda_kernel_launches']} CUDA kernel launches"
            )
            for k, v in r["stages_s"].items():
                print(f"  {k:40s} {v:8.4f} s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
