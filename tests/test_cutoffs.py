"""T8: cutoff controls against the oracle with the same set_realspace_cutoff arguments."""

import numpy as np
import pytest
from ase import Atoms
from conftest import BACKENDS, DEVICES, assert_matches, make_settings, oracle, run, structures


def two_atom_cell() -> Atoms:
    return Atoms(
        "CO", positions=[[0, 0, 0], [1.3, 0.4, 0.2]], cell=np.diag([3.2, 3.6, 3.4]), pbc=True
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cn", [25.0, 40.0])
@pytest.mark.parametrize("disp3", [20.0, 25.0, 40.0])
def test_cutoffs(disp3, cn, backend, device):
    atoms = two_atom_cell()
    ref = oracle(atoms, disp3=disp3, cn=cn)
    settings = make_settings(
        device, backend, three_body_cutoff_bohr=disp3, coordination_cutoff_bohr=cn
    )
    assert_matches(run(atoms, settings), ref)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("width", [2.0, 5.0])
@pytest.mark.parametrize("name", ["two_atom", "dmso_crystal", "dmso_monomer_C2H6OS"])
def test_switching_width(name, width, backend, device):
    atoms = two_atom_cell() if name == "two_atom" else structures()[name]
    disp3 = 20.0 if atoms.pbc.all() else 8.0  # the monomer spans < 8 bohr legs partly
    ref = oracle(atoms, disp3=disp3, cn=25.0, width3=width)
    settings = make_settings(
        device,
        backend,
        three_body_cutoff_bohr=disp3,
        coordination_cutoff_bohr=25.0,
        three_body_cutoff_width_bohr=width,
    )
    assert_matches(run(atoms, settings), ref)
