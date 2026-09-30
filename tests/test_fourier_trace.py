"""FourierATM building blocks: the triple-dipole trace identity, the Bloch-Ewald dipole matrix
and the separable C9 (docs/fourier_atm_plan.md)."""

import itertools
import math

import numpy as np
import pytest
import torch
from conftest import make_fourier_settings, monomer_in_box, names, structures

from torch_atm.fourier import separable_c9
from torch_atm.fourier.cp import fit_c9
from torch_atm.fourier.dipole import (
    EwaldSetup,
    grid_for_cell,
    molecular_dipole_matrix,
    q_grid,
    trace_energy,
)
from torch_atm.fourier.model import ewald_parameters, long_range_energy
from torch_atm.neighbors import brute_force_neighbors

F64 = torch.float64


def brute_force_undamped(pos: torch.Tensor, a: torch.Tensor, mu: torch.Tensor) -> float:
    total = 0.0
    for i, j, k in itertools.combinations(range(pos.shape[0]), 3):
        c9 = float((mu * a[i] * a[j] * a[k]).sum())
        rij, rjk, rki = pos[j] - pos[i], pos[k] - pos[j], pos[i] - pos[k]
        lij, ljk, lki = rij.norm(), rjk.norm(), rki.norm()
        ci = torch.dot(rij, -rki) / (lij * lki)
        cj = torch.dot(-rij, rjk) / (lij * ljk)
        ck = torch.dot(-rjk, rki) / (ljk * lki)
        total += c9 * float((3 * ci * cj * ck + 1) / (lij * ljk * lki) ** 3)
    return total


def test_trace_identity_molecule():
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(8, 3, generator=g, dtype=F64) * 3
    a = torch.rand(8, 2, generator=g, dtype=F64)
    mu = torch.tensor([0.7, -0.2], dtype=F64)
    mat = molecular_dipole_matrix(pos).to(torch.complex128)[None]
    got = float(trace_energy(mat, torch.ones(1, dtype=F64), a, mu))
    assert got == pytest.approx(brute_force_undamped(pos, a, mu), rel=1e-13)


def random_periodic(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    cell = torch.tensor([[9.0, 0.3, 0.0], [0.5, 10.0, 0.2], [0.0, -0.4, 11.0]], dtype=F64)
    pos = torch.rand(n, 3, generator=g, dtype=F64) @ cell
    return pos, cell


def bloch(pos, cell, real_cutoff, q):
    g, k_cut = ewald_parameters(real_cutoff, 1e-14)
    nl = brute_force_neighbors(pos, cell, True, real_cutoff)
    return EwaldSetup(pos, cell, nl, g, k_cut).matrix(torch.tensor(q, dtype=F64))


def test_ewald_splitting_independence():
    """A(q) must not depend on the real/reciprocal split (catches self and k-space terms)."""
    pos, cell = random_periodic()
    q = [0.25, 0.1, -0.3]
    a1, a2 = bloch(pos, cell, 20.0, q), bloch(pos, cell, 30.0, q)
    assert float((a1 - a2).abs().max()) < 1e-14 * float(a1.abs().max()) * 1e3


def test_bloch_hermitian_and_time_reversal():
    pos, cell = random_periodic()
    a = bloch(pos, cell, 20.0, [0.25, 0.1, -0.3])
    b = bloch(pos, cell, 20.0, [-0.25, -0.1, 0.3])
    scale = float(a.abs().max())
    assert float((a - a.conj().T).abs().max()) < 1e-15 * scale * 1e2
    assert float((b - a.conj()).abs().max()) < 1e-15 * scale * 1e2


def test_supercell_identity():
    """Shifted n^3 grid of the cell = the n^3 supercell at its zone-boundary q, per cell."""
    pos, cell = random_periodic()
    g = torch.Generator().manual_seed(1)
    a = torch.rand(pos.shape[0], 2, generator=g, dtype=F64)
    mu = torch.tensor([0.7, -0.2], dtype=F64)
    fractions, weights = q_grid((2, 2, 2), pos.device)
    mats = torch.stack([bloch(pos, cell, 20.0, f.tolist()) for f in fractions])
    per_cell = float(trace_energy(mats, weights, a, mu))
    shifts = torch.tensor(list(itertools.product(range(2), repeat=3)), dtype=F64)
    big = torch.cat([pos + s @ cell for s in shifts])
    sc = bloch(big, 2 * cell, 20.0, [0.5, 0.5, 0.5])[None]
    supercell = float(trace_energy(sc, torch.ones(1, dtype=F64), a.repeat(8, 1), mu)) / 8
    assert per_cell == pytest.approx(supercell, rel=1e-12)


@pytest.mark.parametrize("n", [(1, 1, 1), (2, 2, 2), (3, 2, 1), (4, 4, 3)])
def test_q_grid(n):
    fractions, weights = q_grid(n, torch.device("cpu"))
    assert float(weights.sum()) == pytest.approx(1.0)
    assert fractions.shape[0] == math.prod(n) // 2 + (1 if all(m % 2 for m in n) else 0)
    assert bool(((fractions > 0) & (fractions < 1)).all())


def test_grid_for_cell():
    cell = torch.diag(torch.tensor([20.0, 30.0, 70.0], dtype=F64))
    assert grid_for_cell(cell, 70.0) == (4, 3, 2)


def test_singular_correction_convergence():
    """The q -> 0 correction turns the O(1/N_q) grid error into a much smaller one."""
    atoms = monomer_in_box()
    fs = make_fourier_settings(coordination="d3")
    bohr = fs.bohr_in_angstrom
    cell = torch.tensor(atoms.cell.array)
    a, mu = separable_c9(atoms.numbers, atoms.positions, cell, True, fs)
    pos, cell = torch.tensor(atoms.positions) / bohr, cell / bohr
    nl = brute_force_neighbors(pos, cell, True, 20.0)
    g, k_cut = ewald_parameters(20.0, 1e-10)

    def energy(n, correct):
        return float(long_range_energy(pos, cell, a, mu, nl, g, k_cut, (n, n, n), correct))

    ref = energy(5, True)
    raw_2, raw_3 = abs(energy(2, False) - ref), abs(energy(3, False) - ref)
    cor_2, cor_3 = abs(energy(2, True) - ref), abs(energy(3, True) - ref)
    assert cor_2 < 0.4 * raw_2 and cor_3 < 0.15 * raw_3
    assert raw_2 / raw_3 == pytest.approx(27 / 8, rel=0.15)  # the uncorrected 1/N_q law


SPECIES_SETS = sorted({tuple(sorted(set(structures()[n].numbers.tolist()))) for n in names("")})


@pytest.mark.parametrize("numbers", SPECIES_SETS)
def test_cp_fit(numbers):
    fit = fit_c9(numbers, 2)
    assert fit.max_rel_error < 5e-3
    assert torch.all(fit.factors[0] == 0)  # dummies


@pytest.mark.parametrize("name", ["glycerol_crystal", "emim_ntf2_crystal"])
def test_separable_c9_matches_d3(name):
    """sum_r mu a a a against D3's sqrt(C6 C6 C6) for random atom triples."""
    from torch_atm.atm import pad_positions, pad_system
    from torch_atm.c6 import pairwise_c6, reference_weights
    from torch_atm.coordination import coordination_number
    from torch_atm.data import reference_data

    atoms = structures()[name]
    fs = make_fourier_settings(coordination="d3")
    cell = torch.tensor(atoms.cell.array)
    a, mu = separable_c9(atoms.numbers, atoms.positions, cell, True, fs)
    n = len(atoms)
    data = reference_data(torch.device("cpu"), F64)
    pos, cell_b = torch.tensor(atoms.positions) / fs.bohr_in_angstrom, cell / fs.bohr_in_angstrom
    nl = brute_force_neighbors(pos, cell_b, True, 40.0).pad(n + 1, 8)
    s = pad_system(torch.tensor(atoms.numbers), n + 1, data)
    vec = nl.vectors(pad_positions(pos, n + 1), cell_b)
    cn = coordination_number(vec, nl.mask, s.rcov, s.rcov[nl.indices], 40.0)
    c6 = pairwise_c6(reference_weights(cn, s.ref_cn, s.ref_mask), s.species, s.c6_species)
    rng = np.random.default_rng(0)
    t = torch.tensor(rng.integers(0, n, (4000, 3)))
    i, j, k = t.T
    exact = torch.sqrt(c6[i, j] * c6[j, k] * c6[k, i])
    approx = (mu * a[i] * a[j] * a[k]).sum(-1)
    rel = (approx / exact - 1).abs()
    assert float(rel.max()) < 1e-2 and float(rel.square().mean().sqrt()) < 2e-3
