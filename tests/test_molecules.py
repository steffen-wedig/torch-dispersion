"""T2: molecular ATM energy and gradient against simple-dftd3."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from conftest import (
    BACKENDS,
    COMPILE,
    DEVICES,
    assert_matches,
    make_settings,
    names,
    oracle,
    random_cluster,
    run,
    stored_reference,
    structures,
)


def water_trimer() -> Atoms:
    atoms = Atoms()
    for k in range(3):
        w = molecule("H2O")
        angle = 2 * np.pi * k / 3
        w.rotate(np.degrees(angle) + 30, "z")
        w.translate([2.75 * np.cos(angle), 2.75 * np.sin(angle), 0.0])
        atoms += w
    return atoms


def benzene_dimer() -> Atoms:
    a = molecule("C6H6")
    b = molecule("C6H6")
    b.translate([0.0, 0.0, 3.7])
    b.rotate(30, "z")
    return a + b


EXTRA = {
    "water_trimer": water_trimer,
    "benzene_dimer": benzene_dimer,
    "random_cluster_60": lambda: random_cluster(60, seed=7),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compile", COMPILE)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", names("monomer"))
def test_monomers(name, backend, compile, device):
    result = run(structures()[name], make_settings(device, backend, compile))
    assert_matches(result, stored_reference(name, "default"))
    assert result.stress is None and result.virial is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compile", COMPILE)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(EXTRA))
def test_complexes(name, backend, compile, device):
    atoms = EXTRA[name]()
    ref = oracle(atoms)
    result = run(atoms, make_settings(device, backend, compile))
    assert_matches(result, ref)


def test_monomer_sanity():
    """ATM of a single organic molecule is small and positive (≤ 2 meV, design §9)."""
    for name in names("monomer"):
        e = float(run(structures()[name], make_settings()).energy)
        assert 0 < e < 2e-3, (name, e)
