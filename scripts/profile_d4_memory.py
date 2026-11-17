"""Where the GPU memory of a batched D4 call goes.

For li_diffusion's MD batch shape (15 cells, 5460 atoms, s9 = 0, compiled, nvalchemiops):

1. per stage: memory allocated before and after each stage function, and the peak inside it;
2. at the peak of the whole call: the live allocations (CUDA caching-allocator history), grouped
   by the innermost frame in this package that allocated them.

    uv run python scripts/profile_d4_memory.py [--snapshot out.pickle]
"""

import argparse
import functools
import pickle
from collections import defaultdict

import torch
from profile_d4_stages import md_cells, to_inputs

import torch_dispersion.d4.dispersion as d4
from torch_dispersion.d4 import d4_dispersion_batched
from torch_dispersion.neighbors import NeighborMatrix
from torch_dispersion.settings import D4Settings, NvalchemiNeighbors

GIB = 2**30
STAGES: list[tuple[str, float, float, float]] = []  # (name, before, after, peak) in GiB


def watched(name: str, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        after, peak = torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
        STAGES.append((name, before / GIB, after / GIB, peak / GIB))
        return out

    return wrapper


def instrument() -> None:
    d4.build_flat = watched("neighbour list (60 bohr)", d4.build_flat)
    NeighborMatrix.restrict = watched("restrict", NeighborMatrix.restrict)
    d4.check_coincident = watched("coincidence check", d4.check_coincident)
    d4._graph_stage = watched("graph stage forward", d4._graph_stage)
    d4._two_body = watched("two-body tiles", d4._two_body)
    d4._three_body = watched("ATM tiles", d4._three_body)
    d4.run_tiles = watched("  tile loop", d4.run_tiles)
    d4.close_chain = watched("chain closure (backward)", d4.close_chain)


def live_at_peak(snapshot: dict) -> list[tuple[str, int, int]]:
    """(source frame, bytes, count) of the allocations alive at the peak of the trace."""
    events = [e for trace in snapshot["device_traces"] for e in trace]
    live: dict[int, dict] = {}
    current = peak = 0
    peak_live: dict[int, dict] = {}
    for e in events:
        if e["action"] == "alloc":
            live[e["addr"]] = e
            current += e["size"]
            if current > peak:
                peak, peak_live = current, dict(live)
        elif e["action"] == "free_completed" and e["addr"] in live:
            current -= live.pop(e["addr"])["size"]
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for e in peak_live.values():
        frames = [f for f in e.get("frames", []) if "torch_dispersion" in f["filename"]]
        key = "(outside torch_dispersion)"
        if frames:
            f = frames[0]
            key = f"{f['filename'].split('torch_dispersion/')[-1]}:{f['line']} {f['name']}"
        groups[key][0] += e["size"]
        groups[key][1] += 1
    return sorted(((k, v[0], v[1]) for k, v in groups.items()), key=lambda x: -x[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=None, help="write the allocator snapshot here")
    args = parser.parse_args()
    settings = D4Settings.from_functional(
        "r2scan", atm=False, device="cuda", compile=True, neighbor_backend=NvalchemiNeighbors()
    )
    numbers, positions, cells, counts = to_inputs(md_cells())

    def call():
        return d4_dispersion_batched(numbers, positions, cells, True, counts, settings)

    call()  # compile
    torch.cuda.synchronize()
    instrument()
    base = torch.cuda.memory_allocated()
    torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="python")
    call()
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)

    print(f"{len(counts)} cells, {sum(counts)} atoms; baseline {base / GIB:.2f} GiB")
    print(f"\n{'stage':40s} {'before':>8s} {'after':>8s} {'peak':>8s}  (GiB, allocated)")
    for name, before, after, peak in STAGES:
        print(f"{name:40s} {before:8.2f} {after:8.2f} {peak:8.2f}")
    print("\nlive allocations at the peak of the call, by allocating frame:")
    total = 0
    for key, size, count in live_at_peak(snapshot):
        total += size
        if size >= 0.05 * GIB:
            print(f"  {size / GIB:7.2f} GiB  {count:5d} blocks  {key}")
    print(f"  {total / GIB:7.2f} GiB  total live at peak")
    if args.snapshot:
        with open(args.snapshot, "wb") as f:
            pickle.dump(snapshot, f)


if __name__ == "__main__":
    main()
