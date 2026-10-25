"""D4 reference tables, and the CN -> EEQ charges -> weights -> C6 chain against dftd4's
``get_properties`` (coordination numbers, partial charges, C6, static polarisabilities)."""

import hashlib

import numpy as np
import pytest
import torch
from conftest import BOHR, ION_CHARGES, d4_properties, names, random_cell, random_cluster
from conftest import structures as fixture_structures

from torch_atm.c6 import pairwise_c6
from torch_atm.d4 import eeq
from torch_atm.d4.data import _raw, d4_reference_data, reference_c6
from torch_atm.d4.model import d4_coordination_number, polarizabilities, reference_weights
from torch_atm.neighbors import brute_force_neighbors

# update together with scripts/export_d4_reference_data.py
CHECKSUM = "9cd98e3d7f1748dc5258eba5f4abd206"


def table_checksum() -> str:
    h = hashlib.sha256()
    for key, value in sorted(_raw().items()):
        h.update(key.encode())
        h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()[:32]


def test_checksum():
    assert table_checksum() == CHECKSUM


def test_table_structure():
    d = d4_reference_data(torch.device("cpu"), torch.float64)
    present = d.n_gaussians > 0
    assert torch.equal(present.sum(-1), d.n_ref)
    assert set(d.n_gaussians.unique().tolist()) <= {0, 1, 3}
    assert int(d.n_ref[0]) == 0 and int(d.n_ref[1]) == 2 and int(d.n_ref[6]) == 7
    assert bool((d.alpha_iw >= 0).all()) and not bool(d.alpha_iw[~present].any())
    # EEQ parameters are exact float64 values of the published 8-decimal tables
    assert float(d.eeq_chi[1]) == 1.23695041 and float(d.eeq_rad[8]) == 1.23166285
    c6 = reference_c6(d.alpha_iw[1:4], d.alpha_iw[1:4])
    torch.testing.assert_close(c6, c6.permute(1, 0, 3, 2), rtol=1e-15, atol=0)


def properties(atoms, charge=0.0) -> dict[str, np.ndarray]:
    """CN, charges, C6 and polarisabilities with this package's building blocks."""
    data = d4_reference_data(torch.device("cpu"), torch.float64)
    pbc = bool(atoms.pbc.all())
    z = torch.tensor(atoms.numbers)
    zp = torch.cat([z, z.new_zeros(1)])  # neighbour padding index
    pos = torch.tensor(atoms.positions) / BOHR
    cell = torch.tensor(atoms.cell.array) / BOHR if pbc else None
    rc, en = data.covalent_radius, data.electronegativity

    nl = brute_force_neighbors(pos, cell, pbc, 30.0)
    vec = nl.vectors(pos, cell)
    cn = d4_coordination_number(
        vec, nl.mask, rc[z], rc[zp][nl.indices], en[z], en[zp][nl.indices], 30.0
    )
    nl = nl.restrict(pos, cell, eeq.CN_CUTOFF)
    cn_eeq = eeq.eeq_coordination_number(nl.vectors(pos, cell), nl.mask, rc[z], rc[zp][nl.indices])
    x = eeq.electronegativity_vector(cn_eeq, data.eeq_chi[z], data.eeq_kcnchi[z])
    rad, eta = data.eeq_rad[z], data.eeq_eta[z]
    if pbc:
        pairs = eeq.wigner_seitz_images(pos, cell)
        a = eeq.coulomb_matrix_periodic(pos, cell, rad, eta, eeq.ewald_alpha(cell), pairs)
    else:
        a = eeq.coulomb_matrix_molecular(pos, rad, eta)
    q = eeq.solve_charges(a, x, charge)
    w = reference_weights(
        cn, q, data.reference_cn[z], data.reference_q[z], data.n_gaussians[z], data.zeff[z],
        data.hardness[z],
    )  # fmt: skip
    unique, species = torch.unique(z, return_inverse=True)
    alpha = data.alpha_iw[unique]
    c6 = pairwise_c6(w, species, reference_c6(alpha, alpha))
    pol = polarizabilities(w, data.alpha_iw[z][..., 0])
    return {
        "coordination numbers": cn.numpy(),
        "partial charges": q.numpy(),
        "c6 coefficients": c6.numpy(),
        "polarizabilities": pol.numpy(),
    }


# dftd4 4.3.0's EEQ solve is less precise than 4.1.0's: on the 28-atom all-element cluster
# (cond ~ 50) its charges leave a residual of ~5e-12 in the EEQ equations, against ~3e-15 here
# (and ~1e-14 agreement with 4.1.0), so the charges are compared at 1e-11 absolute and the
# charge-scaled C6 and polarisabilities at 1e-10 relative (the CN is unaffected: 1e-12).
RELATIVE = {"coordination numbers": 1e-12, "c6 coefficients": 1e-10, "polarizabilities": 1e-10}


def assert_properties(atoms, charge=0.0):
    got, ref = properties(atoms, charge), d4_properties(atoms, charge)
    for key, value in ref.items():
        atol = 1e-11 if key == "partial charges" else RELATIVE[key] * np.abs(value).max()
        np.testing.assert_allclose(got[key], value, rtol=0, atol=atol, err_msg=key)
    np.testing.assert_allclose(got["partial charges"].sum(), charge, atol=1e-12)


@pytest.mark.parametrize("name", names("monomer") + names("crystal"))
def test_properties_fixtures(name):
    assert_properties(fixture_structures()[name], ION_CHARGES.get(name, 0.0))


@pytest.mark.parametrize("charge", [0.0, 2.0])
def test_properties_all_elements(charge):
    atoms = random_cluster(28, seed=1)
    assert set(atoms.numbers) == {1, 6, 7, 8, 9, 16, 35}
    assert_properties(atoms, charge)


@pytest.mark.parametrize("seed", [3, 4])
def test_properties_random_cells(seed):
    """Small cells with Wigner-Seitz ties (cubic self images) and unwrapped positions."""
    atoms = random_cell(3, seed=seed, cell_lengths=(5.0, 5.0, 5.0), angles=(90, 90, 90))
    lat = atoms.cell.array
    atoms.positions[0] += 2 * lat[0] - lat[1]
    atoms.positions[1] -= lat[2]
    assert_properties(atoms)


def test_eeq_direct_backward_matches_autograd():
    """The recompute-in-backward direct Coulomb sum equals plain autograd through it."""
    data = d4_reference_data(torch.device("cpu"), torch.float64)
    atoms = random_cell(4, seed=2)
    z = torch.tensor(atoms.numbers)
    pos = (torch.tensor(atoms.positions) / BOHR).requires_grad_()
    cell = (torch.tensor(atoms.cell.array) / BOHR).requires_grad_()
    rad = data.eeq_rad[z]
    alpha = eeq.ewald_alpha(cell)
    pairs = eeq.wigner_seitz_images(pos, cell)
    gam = 1.0 / torch.sqrt(rad[:, None] ** 2 + rad[None, :] ** 2)
    weights = torch.randn(len(atoms), len(atoms), dtype=torch.float64)
    a = eeq._DirectCoulomb.apply(pos, cell, pairs, gam, alpha)
    values = eeq._direct_entries(pos, cell, pairs, gam, alpha, slice(None))
    a_plain = eeq._scatter_symmetric(len(atoms), pairs, values)
    torch.testing.assert_close(a, a_plain, rtol=0, atol=0)
    got = torch.autograd.grad((a * weights).sum(), (pos, cell))
    ref = torch.autograd.grad((a_plain * weights).sum(), (pos, cell))
    for g, r in zip(got, ref, strict=True):
        assert bool(r.abs().sum() > 0)
        torch.testing.assert_close(g, r, rtol=1e-13, atol=1e-16)
