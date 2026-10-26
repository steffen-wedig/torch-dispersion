"""What the D4 chain closure (the backward of the graph stage) spends its time on.

Runs ``d4_dispersion`` (s9 = 0, compiled, nvalchemiops) on one small and one large cell and
profiles only the ``close_chain`` call with torch.profiler: the top operators by self CPU and by
self CUDA time, and the number of operator calls.

    uv run python scripts/profile_d4_chain.py
"""

from typing import Any

import torch
from ase.build import bulk
from ase.io import read
from benchmark_torchsim import FIXTURES
from torch.profiler import ProfilerActivity, profile

import torch_dispersion.d4.dispersion as d4_module
from torch_dispersion.d4 import d4_dispersion
from torch_dispersion.settings import D4Settings, NvalchemiNeighbors

PROFILE: dict[str, Any] = {"on": False, "prof": None}
_close_chain = d4_module.close_chain


def profiled_close_chain(*args, **kwargs):
    if not PROFILE["on"]:
        return _close_chain(*args, **kwargs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        out = _close_chain(*args, **kwargs)
        torch.cuda.synchronize()
    PROFILE["prof"] = prof
    return out


d4_module.close_chain = profiled_close_chain  # ty: ignore[invalid-assignment]


def run(atoms, settings):
    return d4_dispersion(
        torch.tensor(atoms.numbers),
        torch.tensor(atoms.positions),
        torch.tensor(atoms.cell.array),
        True,
        settings,
    )


def main() -> None:
    settings = D4Settings.from_functional(
        "r2scan", atm=False, device="cuda", compile=True, neighbor_backend=NvalchemiNeighbors()
    )
    crystals = {a.info["name"]: a for a in read(FIXTURES / "structures.extxyz", ":")}
    systems = {
        "NaCl 2x2x2 (64 atoms)": bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(2),
        "dmso_crystal 5x5x4": crystals["dmso_crystal"].repeat((5, 5, 4)),
    }
    for name, atoms in systems.items():
        for _ in range(2):
            run(atoms, settings)  # compile and warm up
        PROFILE["on"] = True
        run(atoms, settings)
        PROFILE["on"] = False
        prof = PROFILE["prof"]
        assert prof is not None
        events = prof.key_averages()
        calls = sum(e.count for e in events if e.key.startswith("aten::"))
        print(f"\n===== {name}: {len(atoms)} atoms, {calls} aten calls in close_chain")
        print(events.table(sort_by="self_cpu_time_total", row_limit=25, max_name_column_width=70))
        print(events.table(sort_by="self_cuda_time_total", row_limit=15, max_name_column_width=70))


if __name__ == "__main__":
    main()
