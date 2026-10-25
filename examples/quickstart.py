"""torch-dispersion quick start: DFT-D4 and the D3 ATM term on a molecule and a periodic box.

    uv run python examples/quickstart.py                  # CPU (shortened cutoffs, seconds)
    uv run python examples/quickstart.py --device cuda    # GPU (dftd4's default cutoffs)

Inputs are ASE ``Atoms`` (Å); results are eV, eV/Å and eV/Å^3 in the ASE sign conventions.
"""

import argparse

import numpy as np
import torch
from ase import Atoms, units
from ase.build import molecule
from ase.md.velocitydistribution import thermalize_momenta
from ase.md.verlet import VelocityVerlet

from torch_dispersion import (
    ATMSettings,
    BruteForceNeighbors,
    D4Settings,
    NvalchemiNeighbors,
    TorchATMCalculator,
    TorchD4Calculator,
    d4_dispersion,
)


def water_box(n_side: int = 3, spacing: float = 3.1) -> Atoms:
    """n_side^3 water molecules on a slightly jittered grid in a cubic cell (~1 g/cm^3)."""
    rng = np.random.default_rng(0)
    box = Atoms(cell=np.eye(3) * n_side * spacing, pbc=True)
    for ijk in np.ndindex(n_side, n_side, n_side):
        water = molecule("H2O")
        water.rotate(rng.uniform(0, 360), rng.normal(size=3))
        water.translate((np.array(ijk) + 0.5) * spacing + rng.normal(scale=0.1, size=3))
        box += water
    return box


def make_settings(device: str) -> tuple[D4Settings, ATMSettings]:
    """D4 and D3-ATM settings. On the CPU the cutoffs (bohr) are shortened so that the example
    runs in seconds (the ATM triple sum is the expensive part); on a GPU they are the defaults:
    two-body 60, ATM 40, CN 30 (D4) and 40 / 40 (D3 ATM)."""
    gpu = device == "cuda"
    # nvalchemiops (warp) neighbour lists on the GPU; the brute-force reference list on the CPU
    backend = NvalchemiNeighbors() if gpu else BruteForceNeighbors()
    # r2SCAN-D4: the functional's d4.bj-eeq-atm entry in dftd4's parameters.toml
    d4 = D4Settings.from_functional("r2scan", device=device, compile=gpu, neighbor_backend=backend)
    atm = ATMSettings(device=device, compile=gpu, neighbor_backend=backend)
    if not gpu:
        d4 = d4.model_copy(
            update={
                "two_body_cutoff_bohr": 30.0,
                "three_body_cutoff_bohr": 15.0,
                "coordination_cutoff_bohr": 20.0,
            }
        )
        atm = atm.model_copy(
            update={"three_body_cutoff_bohr": 15.0, "coordination_cutoff_bohr": 20.0}
        )
    return d4, atm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    d4_settings, atm_settings = make_settings(args.device)
    print(
        f"device {args.device}, D4 cutoffs (bohr): two-body {d4_settings.two_body_cutoff_bohr}, "
        f"ATM {d4_settings.three_body_cutoff_bohr}, CN {d4_settings.coordination_cutoff_bohr}"
    )

    # 1. functional API on a molecule: the result also carries the EEQ charges
    benzene = molecule("C6H6")
    result = d4_dispersion(
        torch.tensor(benzene.numbers),
        torch.tensor(benzene.positions),
        cell=None,
        pbc=False,
        settings=d4_settings,
    )
    print("\nbenzene, D4")
    print(f"  energy      {float(result.energy):+.6f} eV")
    print(f"    two-body  {float(result.two_body_energy):+.6f} eV")
    print(f"    ATM       {float(result.three_body_energy):+.6f} eV")
    assert result.forces is not None
    print(f"  max |force| {float(result.forces.norm(dim=1).max()):.2e} eV/Å")
    print(f"  EEQ charges C {float(result.charges[0]):+.3f}, H {float(result.charges[-1]):+.3f}")

    # 2. ASE calculators on a periodic box: energy, forces and stress in one evaluation
    box = water_box()
    box.calc = TorchD4Calculator(d4_settings)
    pressure = -np.mean(box.get_stress()[:3]) / units.GPa * 1e4  # bar
    print(f"\nwater box ({len(box)} atoms), D4")
    print(f"  energy   {box.get_potential_energy():+.4f} eV")
    print(f"  pressure {pressure:+.1f} bar (dispersion only)")

    box.calc = TorchATMCalculator(atm_settings)  # D3 ATM term alone, e.g. next to a D3 two-body
    print(f"water box, D3 ATM: {box.get_potential_energy():+.6f} eV")

    # 3. a few NVE steps with the D4 calculator (in practice: SumCalculator([mlip, d4]))
    box.calc = TorchD4Calculator(d4_settings)
    thermalize_momenta(box, temperature_K=300, rng=np.random.default_rng(1))
    md = VelocityVerlet(box, timestep=0.5 * units.fs)
    print("\nNVE with D4 only (0.5 fs steps)")
    for step in range(0, 30, 5):
        if step:
            md.run(5)
        e_pot, e_kin = box.get_potential_energy(), box.get_kinetic_energy()
        print(f"  step {step:3d}  E_pot {e_pot:+.5f}  E_tot {e_pot + e_kin:+.5f} eV")


if __name__ == "__main__":
    main()
