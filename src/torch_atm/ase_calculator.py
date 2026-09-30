"""ASE calculator for the ATM term (energy in eV, forces in eV/Å, stress in eV/Å^3)."""

import numpy as np
import torch
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from .atm import atm_dispersion, check_pbc
from .fourier import fourier_atm
from .settings import ATMSettings, FourierATMSettings


class TorchATMCalculator(Calculator):
    """Axilrod-Teller-Muto three-body dispersion (simple-dftd3 definition): real space with a
    cutoff (``ATMSettings``) or cutoff-free FourierATM (``FourierATMSettings``).

    Requests for forces or stress compute energy, forces and (with PBC) stress together, so an
    MD step that asks for all three evaluates the term once. Every call passes the current atomic
    numbers and nothing species-dependent is cached across calls, so a calculator reused on a
    structure with different species gives the correct result.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, settings: ATMSettings | FourierATMSettings | None = None, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings if settings is not None else ATMSettings()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        assert atoms is not None
        pbc = check_pbc(atoms.pbc)
        if "stress" in properties and not pbc:
            raise PropertyNotImplementedError("stress requires a fully periodic cell")
        want_grad = any(p in properties for p in ("forces", "stress"))
        args = (
            torch.as_tensor(atoms.numbers),
            torch.as_tensor(atoms.positions),
            torch.as_tensor(np.asarray(atoms.cell)) if pbc else None,
            pbc,
        )
        kwargs = {"compute_forces": want_grad, "compute_stress": want_grad and pbc}
        if isinstance(self.settings, FourierATMSettings):
            result = fourier_atm(*args, self.settings, **kwargs)
        else:
            result = atm_dispersion(*args, self.settings, **kwargs)
        energy = float(result.energy)
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        if result.forces is not None:
            self.results["forces"] = result.forces.detach().cpu().numpy()
        if result.stress is not None:
            self.results["stress"] = full_3x3_to_voigt_6_stress(
                result.stress.detach().cpu().numpy()
            )
