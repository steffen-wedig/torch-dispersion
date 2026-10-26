"""Where the time of the D4 two-body tile loop goes: overhead or GPU work.

For li_diffusion's MD batch shape (15 cells, 5460 atoms, s9 = 0, compiled, nvalchemiops,
energy + forces + stress), traces only the two-body stage with torch.profiler and reports:
its wall time against the summed GPU kernel time, the number of tile calls and CUDA kernel
launches, the evaluated pair slots, and the top operators by CPU and by GPU time; then sweeps
pair_block_size x pair_tile_size (untraced wall time of the stage).

    uv run python scripts/profile_d4_two_body.py
"""

import time
from typing import Any

import torch
from profile_d4_stages import md_cells, to_inputs
from torch.profiler import ProfilerActivity, profile

import torch_dispersion.d4.dispersion as d4
from torch_dispersion.d4 import d4_dispersion_batched
from torch_dispersion.settings import D4Settings, NvalchemiNeighbors

STATE: dict[str, Any] = {"on": False, "calls": 0, "slots": 0}
_two_body = d4._two_body
_get_pair_step_fn = d4.get_pair_step_fn


def counted_step_fn(with_grad: bool, compile: bool):
    step = _get_pair_step_fn(with_grad, compile)

    def counted(acc, c6, pos, cells, numbers, local, t, s):
        if STATE["on"]:
            STATE["calls"] += 1
            STATE["slots"] += t.mask.numel()
        return step(acc, c6, pos, cells, numbers, local, t, s)

    return counted


def traced_two_body(*args, **kwargs):
    if not STATE["on"]:
        return _two_body(*args, **kwargs)
    torch.cuda.synchronize()
    if not STATE.get("trace"):  # wall time without the profiler's own overhead
        t0 = time.perf_counter()
        out = _two_body(*args, **kwargs)
        torch.cuda.synchronize()
        STATE["wall_untraced"] = time.perf_counter() - t0
        return out
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        out = _two_body(*args, **kwargs)
        torch.cuda.synchronize()
        STATE["wall"] = time.perf_counter() - t0
    STATE["prof"] = prof
    return out


d4.get_pair_step_fn = counted_step_fn  # ty: ignore[invalid-assignment]
d4._two_body = traced_two_body  # ty: ignore[invalid-assignment]


SWEEP = [(128, 4096), (256, 4096), (512, 4096), (256, 8192), (1024, 2048)]


def sweep(numbers, positions, cells, counts) -> None:
    """Untraced wall time of the two-body stage for (pair_block_size, pair_tile_size) pairs."""
    print("\npair_block_size x pair_tile_size: two-body wall (ms)")
    for block, tile in SWEEP:
        settings = D4Settings.from_functional(
            "r2scan", atm=False, device="cuda", compile=True,
            neighbor_backend=NvalchemiNeighbors(), pair_block_size=block, pair_tile_size=tile,
        )  # fmt: skip
        for _ in range(2):
            d4_dispersion_batched(numbers, positions, cells, True, counts, settings)
        STATE["on"], STATE["trace"] = True, False
        times = []
        for _ in range(3):
            d4_dispersion_batched(numbers, positions, cells, True, counts, settings)
            times.append(STATE["wall_untraced"])
        STATE["on"] = False
        print(f"  {block:5d} x {tile:5d}: {min(times) * 1e3:7.1f}")


def main() -> None:
    settings = D4Settings.from_functional(
        "r2scan", atm=False, device="cuda", compile=True, neighbor_backend=NvalchemiNeighbors()
    )
    atoms = md_cells()
    numbers, positions, cells, counts = to_inputs(atoms)

    def call():
        return d4_dispersion_batched(numbers, positions, cells, True, counts, settings)

    for _ in range(2):
        call()  # compile and warm up
    STATE["on"] = True
    call()  # untimed by the profiler
    STATE["calls"] = STATE["slots"] = 0
    STATE["trace"] = True
    call()
    STATE["on"] = False
    prof = STATE["prof"]
    events = prof.key_averages()
    gpu_us = sum(e.self_device_time_total for e in events if e.device_type.name == "CUDA")
    launches = sum(e.count for e in events if e.device_type.name == "CUDA")
    calls, slots = STATE["calls"], STATE["slots"]
    print(f"{len(counts)} cells, {sum(counts)} atoms")
    print(f"two-body wall without profiler {STATE['wall_untraced'] * 1e3:.1f} ms")
    print(f"two-body wall {STATE['wall'] * 1e3:.1f} ms, GPU kernels {gpu_us / 1e3:.1f} ms busy")
    print(f"tile calls {calls}, CUDA launches {launches} ({launches / max(calls, 1):.1f} per call)")
    print(f"wall per call {STATE['wall'] / max(calls, 1) * 1e6:.0f} us")
    print(f"pair slots evaluated {slots / 1e6:.1f} M")
    print("\ntop operators by self CPU time:")
    print(events.table(sort_by="self_cpu_time_total", row_limit=20, max_name_column_width=60))
    print("top operators by self CUDA time:")
    print(events.table(sort_by="self_cuda_time_total", row_limit=12, max_name_column_width=60))
    sweep(numbers, positions, cells, counts)


if __name__ == "__main__":
    main()
