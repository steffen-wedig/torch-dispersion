"""T3: small periodic cells (cells smaller than the cutoff) against simple-dftd3."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from conftest import (
    BACKENDS,
    COMPILE,
    DEVICES,
    assert_matches,
    make_settings,
    names,
    oracle,
    random_cell,
    run,
    stored_reference,
    structures,
)

# shortened cutoffs keep the default (fast) tests below a second on CPU
SHORT = {"three_body_cutoff_bohr": 20.0, "coordination_cutoff_bohr": 20.0}


def skewed_cell() -> Atoms:
    atoms = random_cell(4, seed=3)
    cell = atoms.cell.array.copy()
    cell[1] += 0.9 * cell[0]
    cell[2] += 0.7 * cell[0] - 0.6 * cell[1]
    atoms.set_cell(cell, scale_atoms=False)
    return atoms


def two_atom_cell() -> Atoms:
    return Atoms(
        "CO", positions=[[0, 0, 0], [1.3, 0.4, 0.2]], cell=np.diag([3.2, 3.6, 3.4]), pbc=True
    )


SMALL = {
    "fcc_ar": lambda: bulk("Ar", "fcc", a=5.26),
    "two_atom": two_atom_cell,
    "skewed": skewed_cell,
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compile", COMPILE)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", names("crystal"))
def test_crystals_short_cutoff(name, backend, compile, device):
    atoms = structures()[name]
    ref = oracle(atoms, disp3=SHORT["three_body_cutoff_bohr"], cn=SHORT["coordination_cutoff_bohr"])
    assert_matches(run(atoms, make_settings(device, backend, compile, **SHORT)), ref)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compile", COMPILE)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(SMALL))
def test_small_cells(name, backend, compile, device):
    """One-atom cell: only self-image triangles (the 1/6 case); two atoms: the 1/2 cases."""
    atoms = SMALL[name]()
    ref = oracle(atoms)
    result = run(atoms, make_settings(device, backend, compile))
    assert_matches(result, ref)
    assert float(ref["atm_energy"]) != 0.0


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compile", COMPILE)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cutoff_set", ["default", "disp3_25"])
@pytest.mark.parametrize("name", names("crystal"))
def test_crystals_stored(name, cutoff_set, backend, compile, device):
    from conftest import CUTOFF_SETS

    c = CUTOFF_SETS[cutoff_set]
    settings = make_settings(
        device,
        backend,
        compile,
        three_body_cutoff_bohr=c["disp3"],
        coordination_cutoff_bohr=c["cn"],
    )
    assert_matches(run(structures()[name], settings), stored_reference(name, cutoff_set))
