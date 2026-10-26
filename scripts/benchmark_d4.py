"""Benchmark the D4 term on one GPU, with a stage breakdown.

Per liquid box and for the benzamide box replicated up to 8x (256-2048 atoms): wall time and
peak memory of the full D4 term (energy only, and energy + forces + stress), of the run without
the ATM term (s9 = 0: CN, EEQ, two-body), and of the EEQ charges alone (forward, and forward +
backward through ``torch.linalg.solve`` and the Coulomb matrix), which decides whether the
autograd EEQ gradient needs a hand-written replacement. Optionally the CPU oracle (dftd4) time.

    uv run python scripts/benchmark_d4.py --out benchmarks/results/d4_h200.json [--oracle]
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from torch_dispersion.core.batch import Batch
from torch_dispersion.core.padding import pad_positions
from torch_dispersion.d4 import d4_dispersion, eeq
from torch_dispersion.d4.data import d4_reference_data
from torch_dispersion.d4.dispersion import _charges
from torch_dispersion.neighbors import make_builder
from torch_dispersion.settings import BruteForceNeighbors, D4Settings, NvalchemiNeighbors

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
# r2SCAN-D4
PARAMS = {"s8": 0.60187490, "a1": 0.51559235, "a2": 5.77342911}


def timed(repeats: int, fn) -> tuple[float, float]:
    """(median seconds, peak MiB) of ``fn()`` after one warm-up call."""
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times), torch.cuda.max_memory_allocated() / 2**20


def d4_call(atoms, settings, grad: bool):
    return lambda: d4_dispersion(
        torch.tensor(atoms.numbers),
        torch.tensor(atoms.positions),
        torch.tensor(atoms.cell.array),
        True,
        settings,
        compute_forces=grad,
        compute_stress=grad,
    )


def eeq_call(atoms, settings, grad: bool):
    """EEQ charges from positions (CN included), and optionally d(sum q w)/dR by autograd."""
    device = settings.torch_device
    f64 = torch.float64
    data = d4_reference_data(device, f64)
    z = torch.tensor(atoms.numbers, device=device)
    bohr = settings.bohr_in_angstrom
    pos = torch.tensor(atoms.positions, device=device, dtype=f64) / bohr
    cell = torch.tensor(atoms.cell.array, device=device, dtype=f64) / bohr
    nl = make_builder(settings.neighbor_backend)(pos, cell, True, eeq.CN_CUTOFF)
    weights = torch.randn(len(atoms), device=device, dtype=f64)
    rad, eta = data.eeq_rad[z], data.eeq_eta[z]
    chi, kcnchi = data.eeq_chi[z], data.eeq_kcnchi[z]
    total = torch.zeros(1, dtype=f64, device=device)
    batch = Batch.from_counts([len(atoms)], device)
    cells = cell[None]

    def fn():
        leaf = pos.clone().requires_grad_(grad)
        with torch.set_grad_enabled(grad):
            zp = torch.cat([z, z.new_zeros(1)])
            rc = data.covalent_radius
            cn = eeq.eeq_coordination_number(
                nl.vectors(pad_positions(leaf, len(atoms)), cell),
                nl.mask,
                rc[z],
                rc[zp][nl.indices],
            )
            alphas = eeq.ewald_alphas(cells)
            pairs = eeq.wigner_seitz_images(leaf.detach(), cells, batch)
            a_dir = eeq.direct_coulomb(leaf, cells, rad, alphas, pairs, batch, settings.compile)
            q = _charges(
                cn, leaf, cells, a_dir, rad, eta, chi, kcnchi, alphas, total, batch.index,
                batch.mask,
            )  # fmt: skip
            if grad:
                torch.autograd.grad((q * weights).sum(), leaf)

    return fn


def oracle_seconds(atoms, settings) -> float:
    from dftd4.interface import DampingParam, DispersionModel

    bohr = settings.bohr_in_angstrom
    model = DispersionModel(
        atoms.numbers, atoms.positions / bohr, lattice=atoms.cell.array / bohr,
        periodic=np.array([True] * 3),
    )  # fmt: skip
    t = time.perf_counter()
    model.get_dispersion(DampingParam(s6=1.0, s9=1.0, alp=16.0, **PARAMS), grad=True)
    return time.perf_counter() - t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-replicas", type=int, default=8)
    parser.add_argument("--oracle", action="store_true", help="also time dftd4 on the CPU")
    parser.add_argument("--backend", choices=["nvalchemi", "brute_force"], default="nvalchemi")
    args = parser.parse_args()
    structures = {a.info["name"]: a for a in read(FIXTURES / "structures.extxyz", ":")}
    backend = NvalchemiNeighbors() if args.backend == "nvalchemi" else BruteForceNeighbors()
    full = D4Settings(s8=PARAMS["s8"], a1=PARAMS["a1"], a2=PARAMS["a2"], neighbor_backend=backend)
    no_atm = full.model_copy(update={"s9": 0.0})
    rows = []

    def record(**row):
        rows.append(row)
        print(json.dumps(row), flush=True)

    def measure(kind: str, name: str, atoms, repeats: int, oracle: bool):
        row = {"kind": kind, "name": name, "atoms": len(atoms)}
        for label, fn in [
            ("d4_e", d4_call(atoms, full, False)),
            ("d4_efs", d4_call(atoms, full, True)),
            ("no_atm_efs", d4_call(atoms, no_atm, True)),
            ("eeq_q", eeq_call(atoms, full, False)),
            ("eeq_q_grad", eeq_call(atoms, full, True)),
        ]:
            seconds, mib = timed(repeats, fn)
            row[f"seconds_{label}"] = seconds
            row[f"peak_mib_{label}"] = mib
        if oracle:
            row["seconds_oracle_cpu"] = oracle_seconds(atoms, full)
        record(**row)

    for name, atoms in structures.items():
        if "liquid" in name:
            measure("liquid", name, atoms, args.repeats, args.oracle)

    base = structures["benzamide_liquid"]
    for reps in [(2, 1, 1), (2, 2, 1), (2, 2, 2)]:
        atoms = base.repeat(reps)
        if len(atoms) > args.max_replicas * len(base):
            break
        measure("scaling", f"benzamide_liquid x{np.prod(reps)}", atoms, 2, False)

    meta = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "host": platform.node(),
        "settings": full.model_dump(mode="json"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
