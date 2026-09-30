"""FourierATM end to end: exact limits, the brute-force periodic triangle sum, derivatives,
compilation, float32 and the calculator."""

import numpy as np
import pytest
import torch
from ase.build import molecule
from conftest import (
    DEVICES,
    make_fourier_settings,
    make_settings,
    monomer_in_box,
    names,
    run,
    run_fourier,
    structures,
)
from fourier_reference import reference_energy

from torch_atm import TorchATMCalculator

H_POS = 1e-4
H_STRAIN = 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("gradient", ["analytic", "vjp"])
@pytest.mark.parametrize("name", names("monomer"))
def test_molecule_exact_limit(name, gradient, device):
    """Short-range cutoff beyond the molecule: trace - undamped + D3 = D3 without cutoff."""
    atoms = structures()[name]
    fs = make_fourier_settings(
        device,
        coordination="d3",
        short_range_cutoff_bohr=60.0,
        short_range_width_bohr=0.0,
        kernel_gradient=gradient,
    )
    got = run_fourier(atoms, fs)
    ref = run(atoms, make_settings(device, three_body_cutoff_bohr=60.0))
    assert float(got.energy) == pytest.approx(float(ref.energy), rel=1e-10, abs=1e-15)
    torch.testing.assert_close(got.forces, ref.forces, rtol=0, atol=1e-12)
    assert got.stress is None and got.q_grid is None


@pytest.mark.parametrize("r_s", [12.0, 20.0])
def test_molecule_brute_force(r_s):
    """Finite short-range cutoff: against the explicit triangle sum of the same definition."""
    atoms = molecule("C6H6")
    atoms.positions *= 1.3  # legs beyond 12 bohr
    fs = make_fourier_settings(coordination="d3", short_range_cutoff_bohr=r_s)
    got = float(run_fourier(atoms, fs, compute_forces=False).energy)
    assert got == pytest.approx(reference_energy(atoms, fs, cutoff=100.0), rel=1e-11)


@pytest.mark.slow
def test_periodic_brute_force():
    """Sparse periodic box: FourierATM against the real-space triangle sum at 120 bohr (whose
    remaining tail is ~1e-8 eV here); absolute because the total is a ~400x cancellation."""
    atoms = monomer_in_box()
    fs = make_fourier_settings(coordination="d3", q_grid=(4, 4, 4))
    got = float(run_fourier(atoms, fs, compute_forces=False, compute_stress=False).energy)
    ref = reference_energy(atoms, fs, cutoff=120.0)
    assert abs(got - ref) < 1e-6


def periodic_brute_force_short(atoms, fs):
    return float(run_fourier(atoms, fs, compute_forces=False, compute_stress=False).energy)


def test_periodic_brute_force_coarse():
    """Fast variant: default grid rule, 90 bohr reference."""
    atoms = monomer_in_box()
    fs = make_fourier_settings(coordination="d3")
    got = periodic_brute_force_short(atoms, fs)
    assert abs(got - reference_energy(atoms, fs, cutoff=90.0)) < 5e-6


def fd_check(atoms, fs, indices=(0, 3, 7)):
    result = run_fourier(atoms, fs)

    def energy(a):
        return float(run_fourier(a, fs, compute_forces=False, compute_stress=False).energy)

    for i in indices:
        for x in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[i, x] += H_POS
            minus.positions[i, x] -= H_POS
            fd = -(energy(plus) - energy(minus)) / (2 * H_POS)
            assert fd == pytest.approx(float(result.forces[i, x]), abs=1e-8)
    if result.virial is None:
        return
    for a, b in [(0, 0), (1, 2), (2, 1), (0, 1)]:
        eps = np.zeros((3, 3))
        eps[a, b] = H_STRAIN
        e = []
        for sign in (1, -1):
            d = atoms.copy()
            deform = np.eye(3) + sign * eps
            d.set_cell(atoms.cell.array @ deform, scale_atoms=False)
            d.positions = atoms.positions @ deform
            e.append(energy(d))
        assert (e[0] - e[1]) / (2 * H_STRAIN) == pytest.approx(float(result.virial[a, b]), abs=1e-7)


SMALL = {
    "q_grid": (2, 2, 2),
    "short_range_cutoff_bohr": 12.0,
    "ewald_real_cutoff_bohr": 15.0,
}


@pytest.mark.parametrize("gradient", ["analytic", "vjp"])
def test_derivatives_periodic(gradient):
    """Smooth CN, so central differences are exact to O(h^2)."""
    fd_check(structures()["dmso_crystal"], make_fourier_settings(kernel_gradient=gradient, **SMALL))


def test_derivatives_molecule():
    fd_check(
        structures()["glycerol_monomer_C3H8O3"], make_fourier_settings(short_range_cutoff_bohr=8.0)
    )


@pytest.mark.parametrize("name", ["dmso_crystal", "benzamide_crystal"])
def test_gradient_modes_agree(name):
    atoms = structures()[name]
    got = [
        run_fourier(atoms, make_fourier_settings(kernel_gradient=g, **SMALL))
        for g in ("analytic", "vjp")
    ]
    torch.testing.assert_close(got[0].forces, got[1].forces, rtol=0, atol=1e-13)
    torch.testing.assert_close(got[0].virial, got[1].virial, rtol=0, atol=1e-12)


def test_translation_and_force_sum():
    atoms = structures()["dmso_crystal"]
    fs = make_fourier_settings(**SMALL)
    r1 = run_fourier(atoms, fs)
    moved = atoms.copy()
    moved.positions += [0.37, -1.1, 2.3]
    r2 = run_fourier(moved, fs)
    assert float(r1.energy) == pytest.approx(float(r2.energy), rel=1e-11)
    assert float(r1.forces.sum(0).abs().max()) < 1e-11


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["dmso_crystal", "glycerol_monomer_C3H8O3"])
def test_compile_matches_eager(name, device):
    atoms = structures()[name]
    eager = run_fourier(atoms, make_fourier_settings(device, compile=False, **SMALL))
    compiled = run_fourier(atoms, make_fourier_settings(device, compile=True, **SMALL))
    assert float(compiled.energy) == pytest.approx(float(eager.energy), rel=1e-12)
    torch.testing.assert_close(compiled.forces, eager.forces, rtol=0, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["dmso_crystal", "emim_ntf2_crystal"])
def test_float32(name, device):
    """The short-range kernel runs in float32 and cancels ~20x against the (float64) trace, so
    float32 costs more accuracy here than in the real-space ATM (measured: forces 2e-4)."""
    atoms = structures()[name]
    r64 = run_fourier(atoms, make_fourier_settings(device, dtype="float64", **SMALL))
    r32 = run_fourier(atoms, make_fourier_settings(device, dtype="float32", **SMALL))
    assert float(r32.energy) == pytest.approx(float(r64.energy), rel=1e-5)
    scale = float(r64.forces.abs().max())
    assert float((r32.forces - r64.forces).abs().max()) < 1e-3 * scale


def test_calculator():
    atoms = structures()["dmso_crystal"].copy()
    fs = make_fourier_settings(**SMALL)
    atoms.calc = TorchATMCalculator(fs)
    ref = run_fourier(atoms, fs)
    assert atoms.get_potential_energy() == pytest.approx(float(ref.energy), rel=1e-13)
    np.testing.assert_allclose(atoms.get_forces(), ref.forces.numpy(), atol=1e-14)
    assert atoms.get_stress().shape == (6,)


def test_zero_s9_and_empty():
    atoms = structures()["dmso_crystal"]
    r = run_fourier(atoms, make_fourier_settings(s9=0.0))
    assert float(r.energy) == 0.0 and float(r.forces.abs().max()) == 0.0


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("name", names("liquid"))
def test_liquids_against_long_cutoff(name):
    """Default settings (smooth CN aside: D3 CN here) against real-space D3 at 80 bohr, which
    still lacks a ~+0.0013 meV/atom tail. Measured differences: -0.0007 to -0.0044 meV/atom,
    from triangles beyond the short-range cutoff being undamped."""
    atoms = structures()[name]
    got = run_fourier(atoms, make_fourier_settings("cuda", compile=True, coordination="d3"))
    ref = run(
        atoms,
        make_settings("cuda", compile=True, three_body_cutoff_bohr=80.0),
        compute_forces=False,
        compute_stress=False,
    )
    n = len(atoms)
    assert abs(float(got.energy) - float(ref.energy)) / n < 1e-5  # eV/atom = 0.01 meV/atom


@pytest.mark.parametrize("device", DEVICES)
def test_no_recompiles(device):
    """MD-like steps trigger no recompilation, and a different system at most one (the dynamic
    stage 1); the short-range kernel has static shapes like the real-space kernel."""
    from conftest import random_cell
    from torch._dynamo.utils import counters

    def graphs():
        return counters["stats"]["unique_graphs"]

    fs = make_fourier_settings(device, compile=True, **SMALL)
    atoms = random_cell(6, seed=21)
    run_fourier(atoms, fs)
    run_fourier(random_cell(5, seed=2), fs)
    rng = np.random.default_rng(0)
    with torch._dynamo.config.patch(error_on_recompile=True):
        before = graphs()
        for _ in range(10):
            atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
            run_fourier(atoms, fs)
        assert graphs() == before
    run_fourier(structures()["dmso_crystal"], fs)
    run_fourier(structures()["glycerol_crystal"], fs)
    assert graphs() - before <= 1
