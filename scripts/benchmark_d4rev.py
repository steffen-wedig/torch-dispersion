"""Time and verify ORCA's D4rev (wB97M-D4rev, wB97X-D4rev) on molecular batches.

D4rev is dftd4's ``d4`` model with the revised parameters ``wb97m-rev`` / ``wb97x-rev``
(ORCA 6.1.1 manual, Table 3.15). ORCA applies it with the ATM term on (s9 = 1). This script
times the batched TorchSim model (energy and forces, float64) on non-periodic batches, with
and without ATM. It compares every system with dftd4 4.3.0 on the CPU, which looks up the
parameters by name itself, and times that loop as the CPU reference.

Batches:
- ``monomers``: 64 rattled fixture monomers (10-19 atoms, neutral and the two ions);
- ``clusters``: the six fixture liquids cut out as non-periodic ~250-atom clusters;
- ``wide``: 16 random 40-120 atom clusters of 21 elements up to Pt, charges -1..+2.

    uv run --extra torchsim python scripts/benchmark_d4rev.py --out benchmarks/results/d4rev.json
"""

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch_sim as ts
from ase import Atoms
from ase.io import read
from dftd4.interface import DampingParam, DispersionModel
from torch_sim.typing import SystemExtras

from torch_dispersion.calculators.torchsim import D4Model
from torch_dispersion.settings import BruteForceNeighbors, D4Settings, NvalchemiNeighbors

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
ION_CHARGES = {"emim_ntf2_monomer_C6H11N2": 1.0, "emim_ntf2_monomer_C2F6NO4S2": -1.0}
WIDE_ELEMENTS = (1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 26, 29, 30, 35, 46, 53, 78)


def fixture_structures() -> dict[str, Atoms]:
    return {a.info["name"]: a for a in read(FIXTURES / "structures.extxyz", ":")}


def with_charge(atoms: Atoms, charge: float) -> Atoms:
    atoms = atoms.copy()
    atoms.info["charge"] = charge
    return atoms


def monomers(rng) -> list[Atoms]:
    pool = [(n, a) for n, a in fixture_structures().items() if "_monomer" in n]
    out = []
    for i in range(64):
        name, atoms = pool[i % len(pool)]
        atoms = with_charge(atoms, ION_CHARGES.get(name, 0.0))
        atoms.positions += rng.normal(0.0, 0.05, atoms.positions.shape)
        out.append(atoms)
    return out


def clusters(rng) -> list[Atoms]:
    out = []
    for name, atoms in fixture_structures().items():
        if "_liquid" in name:
            atoms = with_charge(atoms, 0.0)
            atoms.pbc = False
            atoms.cell = None
            out.append(atoms)
    return out


def wide(rng) -> list[Atoms]:
    out = []
    for i in range(16):
        n = int(rng.integers(40, 121))
        side = (n / 0.06) ** (1 / 3)
        atoms = Atoms(numbers=rng.choice(WIDE_ELEMENTS, n), positions=rng.uniform(0, side, (n, 3)))
        out.append(with_charge(atoms, float(i % 4 - 1)))
    return out


BATCHES = {"monomers": monomers, "clusters": clusters, "wide": wide}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, repeats: int, device) -> tuple[float, float | None]:
    """(median seconds, peak MiB on CUDA) of ``fn()`` after one warm-up call."""
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
    return statistics.median(times), peak


def dftd4_loop(batch, functional: str, atm: bool, settings: D4Settings):
    """dftd4 energies (eV) and forces (eV/Å) per system, in the settings' units."""
    bohr, hartree = settings.bohr_in_angstrom, settings.hartree_in_ev
    energies, forces = [], []
    for atoms in batch:
        model = DispersionModel(atoms.numbers, atoms.positions / bohr, atoms.info["charge"])
        res = model.get_dispersion(DampingParam(method=functional, atm=atm), grad=True)
        energies.append(float(res["energy"]) * hartree)
        forces.append(-np.asarray(res["gradient"]) * hartree / bohr)
    return np.array(energies), np.concatenate(forces)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--functional", nargs="+", default=["wb97m-rev", "wb97x-rev"])
    parser.add_argument("--batches", nargs="+", choices=list(BATCHES), default=list(BATCHES))
    parser.add_argument("--backend", choices=["nvalchemi", "brute_force"], default="nvalchemi")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    backend = NvalchemiNeighbors() if args.backend == "nvalchemi" else BruteForceNeighbors()
    rows = []
    for batch_name in args.batches:
        batch = BATCHES[batch_name](np.random.default_rng(0))
        sizes = [len(a) for a in batch]
        state = ts.io.atoms_to_state(
            batch, device, torch.float64, system_extras_map={SystemExtras.TOTAL_CHARGE: "charge"}
        )
        for functional in args.functional:
            for atm in (True, False):
                settings = D4Settings.from_functional(
                    functional,
                    atm=atm,
                    device=args.device,
                    compile=not args.no_compile,
                    neighbor_backend=backend,
                )
                model = D4Model(settings, compute_stress=False)
                out = model(state)
                t0 = time.perf_counter()
                e_ref, f_ref = dftd4_loop(batch, functional, atm, settings)
                t_dftd4 = time.perf_counter() - t0
                t_gpu, peak = timed(lambda m=model, st=state: m(st), args.repeats, device)
                row = {
                    "batch": batch_name,
                    "systems": len(sizes),
                    "atoms": sum(sizes),
                    "max_atoms": max(sizes),
                    "functional": functional,
                    "atm": atm,
                    "torch_s": t_gpu,
                    "dftd4_cpu_s": t_dftd4,
                    "speedup": t_dftd4 / t_gpu,
                    "peak_mib": peak,
                    "max_abs_diff_energy_ev": float(
                        np.abs(out["energy"].cpu().numpy() - e_ref).max()
                    ),
                    "max_abs_diff_forces_ev_per_a": float(
                        np.abs(out["forces"].cpu().numpy() - f_ref).max()
                    ),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    meta = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "host": platform.node(),
        "torch": torch.__version__,
        "compile": not args.no_compile,
        "backend": args.backend,
        "repeats": args.repeats,
        "dftd4_omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
