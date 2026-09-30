"""Benchmark FourierATM against the real-space ATM.

Liquids: accuracy against the real-space sum at 80 bohr (energy per atom) and wall time for
energy and for energy + forces + stress. Scaling: the benzamide liquid replicated up to 8x
(256-2048 atoms), FourierATM against real space at 25 and 40 bohr. Writes JSON, prints a table.

    uv run python scripts/benchmark_fourier.py --out benchmarks/results/fourier_h200.json
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from ase.io import read

from torch_atm.atm import atm_dispersion
from torch_atm.fourier import fourier_atm
from torch_atm.settings import ATMSettings, FourierATMSettings

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def timed(repeats: int, *call_args) -> tuple[float, float]:
    """(median seconds, peak MiB) of ``call(*call_args)`` after one warm-up call."""

    def fn():
        return call(*call_args)

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


def call(fn, atoms, settings, grad: bool):
    return fn(
        torch.tensor(atoms.numbers),
        torch.tensor(atoms.positions),
        torch.tensor(atoms.cell.array),
        True,
        settings,
        compute_forces=grad,
        compute_stress=grad,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-replicas", type=int, default=8)
    args = parser.parse_args()
    structures = {a.info["name"]: a for a in read(FIXTURES / "structures.extxyz", ":")}
    rows = []

    def record(**row):
        rows.append(row)
        print(json.dumps(row), flush=True)

    fourier = {c: FourierATMSettings(coordination=c) for c in ("d3", "smooth_cut")}
    real = {c: ATMSettings(three_body_cutoff_bohr=c) for c in (25.0, 40.0)}
    for name, atoms in structures.items():
        if "liquid" not in name:
            continue
        n = len(atoms)
        e80 = float(
            call(atm_dispersion, atoms, ATMSettings(three_body_cutoff_bohr=80.0), False).energy
        )
        for label, fn, settings in [
            ("fourier_d3cn", fourier_atm, fourier["d3"]),
            ("fourier_smoothcn", fourier_atm, fourier["smooth_cut"]),
            ("real_25", atm_dispersion, real[25.0]),
            ("real_40", atm_dispersion, real[40.0]),
        ]:
            energy = float(call(fn, atoms, settings, False).energy)
            t_e, _ = timed(args.repeats, fn, atoms, settings, False)
            t_efs, mib = timed(args.repeats, fn, atoms, settings, True)
            record(
                kind="liquid",
                name=name,
                atoms=n,
                method=label,
                mev_per_atom=1000 * energy / n,
                minus_real80_mev_per_atom=1000 * (energy - e80) / n,
                seconds_energy=t_e,
                seconds_efs=t_efs,
                peak_mib=mib,
            )

    base = structures["benzamide_liquid"]
    for reps in [(1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2)]:
        atoms = base.repeat(reps)
        if len(atoms) > args.max_replicas * len(base):
            break
        for label, fn, settings in [
            ("fourier_smoothcn", fourier_atm, fourier["smooth_cut"]),
            ("real_25", atm_dispersion, real[25.0]),
            ("real_40", atm_dispersion, real[40.0]),
        ]:
            t_efs, mib = timed(max(2, args.repeats // 2), fn, atoms, settings, True)
            record(kind="scaling", atoms=len(atoms), method=label, seconds_efs=t_efs, peak_mib=mib)

    meta = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "host": platform.node(),
        "fourier_settings": fourier["smooth_cut"].model_dump(mode="json"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
