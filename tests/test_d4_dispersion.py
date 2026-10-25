"""DFT-D4 energy, gradient and virial against dftd4 4.3.0.

Default tests run molecules at dftd4's default cutoffs and periodic cells at shortened cutoffs
(``set_realspace_cutoff``), which keeps the CPU triangle sums small; the default 40 bohr ATM
cutoff on periodic cells is ``slow`` (run on a GPU).
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from conftest import (
    BACKENDS,
    COMPILE,
    DEVICES,
    ION_CHARGES,
    TOL_ENERGY,
    TOL_GRADIENT,
    TOL_VIRIAL,
    d4_cutoff_settings,
    d4_oracle,
    make_d4_settings,
    names,
    random_cluster,
    run_d4,
    structures,
    to_atomic_units,
)

# (disp2, disp3, cn) in bohr
SHORT = (30.0, 20.0, 20.0)


def assert_matches(result, ref, rtol_energy=0.0):
    got = to_atomic_units(result)
    e_ref = float(ref["energy"])
    assert abs(got.energy - e_ref) <= max(TOL_ENERGY, rtol_energy * abs(e_ref)), (got.energy, e_ref)
    if got.gradient is not None:
        np.testing.assert_allclose(got.gradient, ref["gradient"], rtol=0, atol=TOL_GRADIENT)
    if got.virial is not None:
        np.testing.assert_allclose(got.virial, ref["virial"], rtol=0, atol=TOL_VIRIAL)


def two_atom_cell() -> Atoms:
    return Atoms(
        "CO", positions=[[0, 0, 0], [1.3, 0.4, 0.2]], cell=np.diag([3.2, 3.6, 3.4]), pbc=True
    )


SMALL = {"fcc_ar": lambda: bulk("Ar", "fcc", a=5.26), "two_atom": two_atom_cell}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compile", COMPILE)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", names("monomer"))
def test_monomers(name, backend, compile, device):
    atoms = structures()[name]
    settings = make_d4_settings(device, backend, compile)
    assert_matches(run_d4(atoms, settings), d4_oracle(atoms))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", list(ION_CHARGES))
def test_charged_ions(name, device):
    """The total charge enters the EEQ constraint (and through q the two-body C6)."""
    atoms, charge = structures()[name], ION_CHARGES[name]
    result = run_d4(atoms, make_d4_settings(device), charge=charge)
    assert_matches(result, d4_oracle(atoms, charge=charge))
    np.testing.assert_allclose(float(result.charges.sum()), charge, atol=1e-12)
    neutral = run_d4(atoms, make_d4_settings(device))
    assert abs(float(result.two_body_energy - neutral.two_body_energy)) > 1e-6
    assert float(result.three_body_energy) == pytest.approx(float(neutral.three_body_energy))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_random_cluster(backend, device):
    """60 atoms, every fixture element, 1.0 Å close contacts."""
    atoms = random_cluster(60, seed=7)
    assert_matches(run_d4(atoms, make_d4_settings(device, backend)), d4_oracle(atoms))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compile", COMPILE)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", names("crystal"))
def test_crystals_short_cutoff(name, backend, compile, device):
    atoms = structures()[name]
    settings = make_d4_settings(device, backend, compile, **d4_cutoff_settings(*SHORT))
    assert_matches(run_d4(atoms, settings), d4_oracle(atoms, cutoffs=SHORT))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(SMALL))
def test_small_cells_short_cutoff(name, backend, device):
    """One atom: only self-image triangles (the 1/6 case); two atoms: the 1/2 cases."""
    atoms = SMALL[name]()
    settings = make_d4_settings(device, backend, **d4_cutoff_settings(*SHORT))
    result = run_d4(atoms, settings)
    assert_matches(result, d4_oracle(atoms, cutoffs=SHORT))
    assert float(result.three_body_energy) != 0.0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cutoffs", [(25.0, 15.0, 15.0), (40.0, 25.0, 30.0), (20.0, 20.0, 40.0)])
def test_cutoff_controls(cutoffs, device):
    """Each cutoff on its own, against the oracle with the same ``set_realspace_cutoff``."""
    atoms = structures()["dmso_crystal"]
    settings = make_d4_settings(device, **d4_cutoff_settings(*cutoffs))
    assert_matches(run_d4(atoms, settings), d4_oracle(atoms, cutoffs=cutoffs))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", names("crystal"))
def test_crystals_two_body(name, backend, device):
    """s9 = 0 at the default cutoffs: two-body term with the full CN and charge chains."""
    atoms = structures()[name]
    result = run_d4(atoms, make_d4_settings(device, backend, s9=0.0))
    assert_matches(result, d4_oracle(atoms, s9=0.0))
    assert float(result.three_body_energy) == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compile", COMPILE)
@pytest.mark.parametrize("name", names("crystal"))
def test_crystals(name, compile, device):
    atoms = structures()[name]
    assert_matches(run_d4(atoms, make_d4_settings(device, compile=compile)), d4_oracle(atoms))


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", list(SMALL))
def test_small_cells(name, device):
    atoms = SMALL[name]()
    assert_matches(run_d4(atoms, make_d4_settings(device)), d4_oracle(atoms))
