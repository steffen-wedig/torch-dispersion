"""D4 forces and virial against central finite differences of this package's own energy, the
charge chain on its own, and torch.compile: compiled equals eager and MD-like steps do not
recompile."""

import numpy as np
import pytest
import torch
from conftest import DEVICES, make_d4_settings, random_cell, random_cluster, run_d4, structures
from torch._dynamo.utils import counters

import torch_dispersion.d4.dispersion as d4_module

H_POS = 2.5e-5  # Å (close contacts: the FD truncation error is ~3e-9 at 1e-4)
H_STRAIN = 1e-5
SHORT = {
    "two_body_cutoff_bohr": 25.0,
    "three_body_cutoff_bohr": 16.0,
    "coordination_cutoff_bohr": 18.0,
}

SYSTEMS = {
    "cluster": lambda: random_cluster(12, seed=5, density=0.05),
    "cell": lambda: random_cell(3, seed=11),
}


def energy(atoms, settings, charge=0.0) -> float:
    result = run_d4(atoms, settings, charge, compute_forces=False, compute_stress=False)
    return float(result.energy)


def fd_forces(atoms, settings, indices, charge=0.0) -> np.ndarray:
    out = np.zeros((len(indices), 3))
    for k, i in enumerate(indices):
        for x in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[i, x] += H_POS
            minus.positions[i, x] -= H_POS
            e = energy(plus, settings, charge) - energy(minus, settings, charge)
            out[k, x] = -e / (2 * H_POS)
    return out


def fd_virial(atoms, settings) -> np.ndarray:
    out = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            eps = np.zeros((3, 3))
            eps[a, b] = H_STRAIN
            e = []
            for sign in (1, -1):
                deformed = atoms.copy()
                deform = np.eye(3) + sign * eps
                deformed.set_cell(atoms.cell.array @ deform, scale_atoms=False)
                deformed.positions = atoms.positions @ deform
                e.append(energy(deformed, settings))
            out[a, b] = (e[0] - e[1]) / (2 * H_STRAIN)
    return out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("system", list(SYSTEMS))
def test_forces_fd(system, device):
    atoms = SYSTEMS[system]()
    settings = make_d4_settings(device, **SHORT)
    forces = run_d4(atoms, settings, charge=0.5).forces.cpu().numpy()
    idx = [0, len(atoms) // 2, len(atoms) - 1]
    fd = fd_forces(atoms, settings, idx, charge=0.5)
    np.testing.assert_allclose(forces[idx], fd, rtol=0, atol=2e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_virial_fd(device):
    atoms = SYSTEMS["cell"]()
    settings = make_d4_settings(device, **SHORT)
    result = run_d4(atoms, settings)
    virial = result.virial.cpu().numpy()
    np.testing.assert_allclose(virial, fd_virial(atoms, settings), rtol=0, atol=2e-8)
    volume = abs(np.linalg.det(atoms.cell.array))
    np.testing.assert_allclose(result.stress.cpu().numpy(), 0.5 * (virial + virial.T) / volume)


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_charge_chain_contribution(system, monkeypatch):
    """With the EEQ charges frozen the forces lose a non-negligible part, and the frozen-charge
    forces equal the FD of the frozen-charge energy."""
    atoms = SYSTEMS[system]()
    settings = make_d4_settings(**SHORT)
    full = run_d4(atoms, settings)
    store = {}
    charges = d4_module._charges

    def frozen(*args, **kwargs):
        q = charges(*args, **kwargs)
        store.setdefault("q", q.detach().clone())
        return store["q"]

    monkeypatch.setattr(d4_module, "_charges", frozen)
    explicit = run_d4(atoms, settings)
    idx = [0, len(atoms) - 1]
    np.testing.assert_allclose(
        explicit.forces.cpu().numpy()[idx], fd_forces(atoms, settings, idx), rtol=0, atol=2e-9
    )
    assert torch.equal(full.energy, explicit.energy)
    assert (full.forces - explicit.forces).abs().max() > 1e-4 * full.forces.abs().max()


def unique_graphs() -> int:
    return counters["stats"]["unique_graphs"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["dmso_crystal", "glycerol_monomer_C3H8O3"])
def test_compiled_equals_eager(name, device):
    atoms = structures()[name]
    eager = run_d4(atoms, make_d4_settings(device, compile=False, **SHORT))
    compiled = run_d4(atoms, make_d4_settings(device, compile=True, **SHORT))
    torch.testing.assert_close(compiled.energy, eager.energy, rtol=1e-14, atol=0)
    torch.testing.assert_close(compiled.forces, eager.forces, rtol=0, atol=1e-14)
    if eager.stress is not None:
        torch.testing.assert_close(compiled.virial, eager.virial, rtol=0, atol=1e-13)


@pytest.mark.parametrize("device", DEVICES)
def test_no_recompiles_in_md(device):
    """50 MD-like steps (small displacements) trigger zero recompilations."""
    atoms = random_cell(6, seed=21)
    settings = make_d4_settings(device, compile=True, **SHORT)
    run_d4(atoms, settings)
    run_d4(atoms, settings)  # stage-1 graphs settle on dynamic shapes
    rng = np.random.default_rng(0)
    with torch._dynamo.config.patch(error_on_recompile=True):
        before = unique_graphs()
        for _ in range(50):
            atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
            run_d4(atoms, settings)
        assert unique_graphs() == before
