"""T11: edge cases."""

import numpy as np
import pytest
import torch
from ase import Atoms
from conftest import DEVICES, make_settings, oracle, run, structures, to_atomic_units

from torch_atm.atm import atm_dispersion
from torch_atm.c6 import reference_weights
from torch_atm.data import reference_data
from torch_atm.settings import ATMSettings


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["dmso_crystal", "dmso_monomer_C2H6OS"])
def test_s9_zero(name, device):
    atoms = structures()[name]
    result = run(atoms, make_settings(device, s9=0.0))
    assert float(result.energy) == 0.0
    assert not result.forces.any()
    if atoms.pbc.all():
        assert not result.stress.any() and not result.virial.any()
    else:
        assert result.stress is None


@pytest.mark.parametrize("device", DEVICES)
def test_single_atom(device):
    result = run(Atoms("C", positions=[[0, 0, 0]]), make_settings(device))
    assert float(result.energy) == 0.0 and result.forces.shape == (1, 3)


@pytest.mark.parametrize("device", DEVICES)
def test_two_atoms_no_triangles(device):
    result = run(Atoms("CO", positions=[[0, 0, 0], [1.2, 0, 0]]), make_settings(device))
    assert float(result.energy) == 0.0 and not result.forces.any()


@pytest.mark.parametrize("device", DEVICES)
def test_empty_neighbor_rows(device):
    """An isolated atom far from a molecule has no neighbours and no effect."""
    mol = structures()["dmso_monomer_C2H6OS"]
    far = mol + Atoms("Ne", positions=[[200.0, 0.0, 0.0]])
    ref = run(mol, make_settings(device))
    got = run(far, make_settings(device))
    torch.testing.assert_close(got.energy, ref.energy, rtol=1e-14, atol=0)
    assert not got.forces[-1].any()
    oracle_far = to_atomic_units(got)
    assert abs(oracle_far.energy - float(oracle(far)["atm_energy"])) < 1e-10


@pytest.mark.parametrize("device", DEVICES)
def test_coincident_atoms_raise(device):
    with pytest.raises(ValueError, match="coincident"):
        run(Atoms("CCO", positions=[[0, 0, 0], [0, 0, 0], [1.2, 0, 0]]), make_settings(device))


def test_coincident_periodic_image_raises():
    atoms = Atoms("CO", positions=[[0, 0, 0], [1.0, 1.0, 1.0]], cell=np.eye(3) * 4, pbc=True)
    atoms.positions[1] = atoms.positions[0] + atoms.cell[0]
    with pytest.raises(ValueError, match="coincident"):
        run(atoms, make_settings())


def test_mixed_pbc_raises():
    with pytest.raises(ValueError, match="mixed periodicity"):
        atm_dispersion(
            torch.tensor([6, 8]),
            torch.tensor([[0.0, 0, 0], [1.2, 0, 0]]),
            torch.eye(3) * 5,
            [True, True, False],
            make_settings(),
        )


def test_settings_validation():
    with pytest.raises(ValueError):
        ATMSettings(three_body_cutoff_bohr=-1.0)
    with pytest.raises(ValueError):
        ATMSettings(three_body_cutoff_bohr=20.0, three_body_cutoff_width_bohr=20.0)
    with pytest.raises(ValueError):
        ATMSettings(coordination_cutoff_bohr=0.0)
    with pytest.raises(ValueError):
        ATMSettings(three_body_cutoff_width_bohr=-0.1)


@pytest.mark.parametrize("device", DEVICES)
def test_exceptional_weights(device):
    """A CN far beyond every reference underflows the normalisation: the weights must be
    finite, one-hot on the largest reference CN, with zero derivative."""
    data = reference_data(torch.device(device), torch.float64)
    z = torch.tensor([6, 1, 8], device=device)
    ref_cn = data.reference_cn[z]
    ref_mask = torch.arange(7, device=device) < data.n_ref[z][:, None]
    cn = torch.tensor([60.0, 50.0, 2.0], device=device, dtype=torch.float64, requires_grad=True)
    w = reference_weights(cn, ref_cn, ref_mask)
    assert torch.isfinite(w).all()
    for row in range(2):
        largest = torch.where(ref_mask[row], ref_cn[row], -torch.inf).argmax()
        expected = torch.zeros(7, dtype=torch.float64, device=device)
        expected[largest] = 1.0
        torch.testing.assert_close(w[row], expected, rtol=0, atol=0)
    grad = torch.autograd.grad(w[:2].sum() + (w[:2] * torch.arange(7, device=device)).sum(), cn)[0]
    assert grad[0] == 0.0 and grad[1] == 0.0
    # the regular atom still has normalised weights and a finite derivative
    torch.testing.assert_close(w[2].sum(), torch.ones((), dtype=torch.float64, device=device))
    assert torch.isfinite(grad[2])
