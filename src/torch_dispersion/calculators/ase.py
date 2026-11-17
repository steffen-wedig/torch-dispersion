"""ASE calculators for the ATM term and DFT-D4 (energy in eV, forces in eV/Å, stress in eV/Å^3)."""

import numpy as np
import torch
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from ..atm import atm_dispersion
from ..core.inputs import check_pbc
from ..d4 import d4_dispersion
from ..settings import ATMSettings, D4Settings


class ThreeBodyCalculator(Calculator):
    """ASE plumbing of the three-body terms; subclasses set ``settings`` and ``_evaluate``.

    Requests for forces or stress compute energy, forces and (with PBC) stress together, so an
    MD step that asks for all three evaluates the term once. Every call passes the current atomic
    numbers and nothing species-dependent is cached across calls, so a calculator reused on a
    structure with different species gives the correct result.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def _evaluate(
        self, numbers, positions, cell, pbc: bool, compute_forces: bool, compute_stress: bool
    ):
        raise NotImplementedError

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        assert atoms is not None
        pbc = check_pbc(atoms.pbc)
        if "stress" in properties and not pbc:
            raise PropertyNotImplementedError("stress requires a fully periodic cell")
        want_grad = any(p in properties for p in ("forces", "stress"))
        result = self._evaluate(
            torch.as_tensor(atoms.numbers),
            torch.as_tensor(atoms.positions),
            torch.as_tensor(np.asarray(atoms.cell)) if pbc else None,
            pbc,
            compute_forces=want_grad,
            compute_stress=want_grad and pbc,
        )
        energy = float(result.energy)
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        if result.forces is not None:
            self.results["forces"] = result.forces.detach().cpu().numpy()
        if result.stress is not None:
            self.results["stress"] = full_3x3_to_voigt_6_stress(
                result.stress.detach().cpu().numpy()
            )


class TorchATMCalculator(ThreeBodyCalculator):
    """Axilrod-Teller-Muto three-body dispersion (simple-dftd3 definition), real space with a
    cutoff (``ATMSettings``). The experimental cutoff-free variant is
    ``torch_dispersion.experimental.fourier_atm.FourierATMCalculator``."""

    def __init__(self, settings: ATMSettings | None = None, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings if settings is not None else ATMSettings()

    def _evaluate(
        self, numbers, positions, cell, pbc: bool, compute_forces: bool, compute_stress: bool
    ):
        return atm_dispersion(
            numbers, positions, cell, pbc, self.settings, compute_forces, compute_stress
        )


class TorchD4Calculator(Calculator):
    """DFT-D4 dispersion (dftd4 4.3.0 ``d4`` model with EEQ charges): two-body rational damping
    plus ATM. ``charge`` is the total charge of the structures it is used on.

    Like ``TorchATMCalculator``, forces or stress requests compute energy, forces and (with PBC)
    stress together, and nothing species-dependent is cached across calls.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress", "charges"]

    def __init__(self, settings: D4Settings, charge: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings
        self.charge = charge

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        assert atoms is not None
        pbc = check_pbc(atoms.pbc)
        if "stress" in properties and not pbc:
            raise PropertyNotImplementedError("stress requires a fully periodic cell")
        want_grad = any(p in properties for p in ("forces", "stress"))
        result = d4_dispersion(
            torch.as_tensor(atoms.numbers),
            torch.as_tensor(atoms.positions),
            torch.as_tensor(np.asarray(atoms.cell)) if pbc else None,
            pbc,
            self.settings,
            charge=self.charge,
            compute_forces=want_grad,
            compute_stress=want_grad and pbc,
        )
        energy = float(result.energy)
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["charges"] = result.charges.detach().cpu().numpy()
        if result.forces is not None:
            self.results["forces"] = result.forces.detach().cpu().numpy()
        if result.stress is not None:
            self.results["stress"] = full_3x3_to_voigt_6_stress(
                result.stress.detach().cpu().numpy()
            )
