"""T1: the CN -> weights -> C6 chain, validated through the D3(BJ) two-body energy."""

import pytest
import torch
from conftest import (
    BACKENDS,
    DAMPING,
    DEVICES,
    HARTREE,
    TOL_ENERGY,
    make_settings,
    names,
    oracle,
    random_cluster,
    stored_reference,
    structures,
)

from torch_atm.two_body import two_body_energy


def two_body(atoms, settings) -> float:
    pbc = bool(atoms.pbc.all())
    e = two_body_energy(
        torch.tensor(atoms.numbers),
        torch.tensor(atoms.positions),
        torch.tensor(atoms.cell.array) if pbc else None,
        pbc,
        settings,
        s6=DAMPING["s6"],
        s8=DAMPING["s8"],
        a1=DAMPING["a1"],
        a2=DAMPING["a2"],
    )
    return float(e) / HARTREE


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", names("monomer"))
def test_monomers(name, backend, device):
    ref = stored_reference(name, "default")["two_body_energy"]
    got = two_body(structures()[name], make_settings(device, backend))
    assert abs(got - float(ref)) <= TOL_ENERGY


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", names("crystal"))
def test_crystals(name, backend, device):
    ref = stored_reference(name, "default")["two_body_energy"]
    got = two_body(structures()[name], make_settings(device, backend))
    assert abs(got - float(ref)) <= TOL_ENERGY


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_random_cluster_all_elements(backend, device):
    atoms = random_cluster(28, seed=1)
    assert set(atoms.numbers) == {1, 6, 7, 8, 9, 16, 35}
    ref = oracle(atoms, grad=False)["two_body_energy"]
    got = two_body(atoms, make_settings(device, backend))
    assert abs(got - float(ref)) <= TOL_ENERGY
