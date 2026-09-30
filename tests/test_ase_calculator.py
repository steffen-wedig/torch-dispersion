"""T12: ASE calculator."""

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase.stress import full_3x3_to_voigt_6_stress
from conftest import DEVICES, make_settings, run, structures

from torch_atm.ase_calculator import TorchATMCalculator

SHORT = {"three_body_cutoff_bohr": 20.0, "coordination_cutoff_bohr": 20.0}


class ConstantCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(self.atoms)
        self.results = {
            "energy": 1.5,
            "free_energy": 1.5,
            "forces": np.full((n, 3), 0.25),
            "stress": np.full(6, 0.01),
        }


def expected(atoms, settings):
    r = run(atoms, settings)
    stress = None if r.stress is None else full_3x3_to_voigt_6_stress(r.stress.cpu().numpy())
    return float(r.energy), r.forces.cpu().numpy(), stress


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["dmso_crystal", "dmso_monomer_C2H6OS"])
def test_properties_match_functional_api(name, device):
    settings = make_settings(device, **SHORT)
    atoms = structures()[name].copy()
    atoms.calc = TorchATMCalculator(settings)
    e, f, s = expected(atoms, settings)
    assert atoms.get_potential_energy() == pytest.approx(e, rel=1e-14)
    np.testing.assert_allclose(atoms.get_forces(), f, rtol=0, atol=1e-14)
    if s is not None:
        np.testing.assert_allclose(atoms.get_stress(), s, rtol=0, atol=1e-14)


def test_species_change_on_reused_calculator():
    settings = make_settings(**SHORT)
    calc = TorchATMCalculator(settings)
    a = structures()["dmso_crystal"].copy()
    a.calc = calc
    e_a = a.get_potential_energy()
    b = a.copy()
    b.numbers = np.where(b.numbers == 16, 8, b.numbers)  # S -> O, same geometry
    b.calc = calc
    assert b.get_potential_energy() == pytest.approx(expected(b, settings)[0], rel=1e-14)
    assert b.get_potential_energy() != pytest.approx(e_a, rel=1e-6)


def test_sum_calculator():
    settings = make_settings(**SHORT)
    atoms = structures()["dmso_crystal"].copy()
    atoms.calc = SumCalculator([TorchATMCalculator(settings), ConstantCalculator()])
    e, f, s = expected(atoms, settings)
    assert atoms.get_potential_energy() == pytest.approx(e + 1.5, rel=1e-14)
    np.testing.assert_allclose(atoms.get_forces(), f + 0.25, rtol=0, atol=1e-14)
    np.testing.assert_allclose(atoms.get_stress(), s + 0.01, rtol=0, atol=1e-14)


def test_molecule_stress_not_implemented():
    from ase.calculators.calculator import PropertyNotImplementedError

    atoms = structures()["dmso_monomer_C2H6OS"].copy()
    atoms.calc = TorchATMCalculator(make_settings(**SHORT))
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_stress()
