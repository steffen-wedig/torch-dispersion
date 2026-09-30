"""T5: forces and virial against central finite differences of this package's own energy,
with and without the CN -> C6 chain."""

import numpy as np
import pytest
import torch
from conftest import BACKENDS, DEVICES, make_settings, random_cell, random_cluster, run

import torch_atm.atm as atm_module

H_POS = 1e-4  # Å
H_STRAIN = 1e-5
SHORT = {"three_body_cutoff_bohr": 18.0, "coordination_cutoff_bohr": 18.0}


def energy(atoms, settings) -> float:
    return float(run(atoms, settings, compute_forces=False, compute_stress=False).energy)


def fd_forces(atoms, settings, indices) -> np.ndarray:
    out = np.zeros((len(indices), 3))
    for k, i in enumerate(indices):
        for x in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[i, x] += H_POS
            minus.positions[i, x] -= H_POS
            out[k, x] = -(energy(plus, settings) - energy(minus, settings)) / (2 * H_POS)
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


@pytest.fixture
def frozen_c6(monkeypatch):
    """Replace stage 1 by a constant C6 taken at the first call (CN held fixed)."""
    store = {}

    def fake_stage1_fn(compile):
        def fn(*args):
            cn, c6 = atm_module._stage1(*args)
            if "c6" not in store:
                store["c6"] = c6.detach().clone()
            return cn.detach(), store["c6"]

        return fn

    monkeypatch.setattr(atm_module, "stage1_fn", fake_stage1_fn)
    return store


SYSTEMS = {
    "cluster": lambda: random_cluster(12, seed=5, density=0.05),
    "cell": lambda: random_cell(3, seed=11),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("system", list(SYSTEMS))
def test_forces_fd(system, backend, device):
    atoms = SYSTEMS[system]()
    settings = make_settings(device, backend, **SHORT)
    forces = run(atoms, settings).forces.cpu().numpy()
    idx = [0, len(atoms) // 2, len(atoms) - 1]
    np.testing.assert_allclose(forces[idx], fd_forces(atoms, settings, idx), rtol=0, atol=2e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_virial_fd(backend, device):
    atoms = SYSTEMS["cell"]()
    settings = make_settings(device, backend, **SHORT)
    result = run(atoms, settings)
    virial = result.virial.cpu().numpy()
    np.testing.assert_allclose(virial, fd_virial(atoms, settings), rtol=0, atol=2e-8)
    volume = abs(np.linalg.det(atoms.cell.array))
    np.testing.assert_allclose(result.stress.cpu().numpy(), 0.5 * (virial + virial.T) / volume)


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_cn_chain_contribution(system, frozen_c6):
    """With C6 frozen the gradient is the explicit part only, and it must match the FD of the
    frozen-C6 energy; the full gradient differs from it by a non-negligible CN part."""
    atoms = SYSTEMS[system]()
    settings = make_settings(**SHORT)
    explicit = run(atoms, settings)
    idx = [0, len(atoms) - 1]
    np.testing.assert_allclose(
        explicit.forces.cpu().numpy()[idx], fd_forces(atoms, settings, idx), rtol=0, atol=2e-9
    )
    if atoms.pbc.all():
        np.testing.assert_allclose(
            explicit.virial.cpu().numpy(), fd_virial(atoms, settings), rtol=0, atol=2e-8
        )
    frozen_c6.clear()
    # a fresh, unfrozen evaluation for comparison
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(atm_module, "stage1_fn", lambda compile: atm_module._stage1)
        full = run(atoms, settings)
    cn_part = (full.forces - explicit.forces).abs().max()
    assert cn_part > 1e-3 * full.forces.abs().max()
    assert torch.equal(full.energy, explicit.energy)
