"""T6: rigid motions, permutation, wrapping and supercell consistency."""

import numpy as np
import pytest
import torch
from ase.build import make_supercell
from conftest import BACKENDS, DEVICES, make_settings, run, structures
from scipy.spatial.transform import Rotation

SHORT = {"three_body_cutoff_bohr": 20.0, "coordination_cutoff_bohr": 20.0}
MOLECULE = "glycerol_monomer_C3H8O3"
CRYSTAL = "dmso_crystal"


def close(a, b, rtol=1e-12, atol=1e-14):
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_rigid_motion(backend, device):
    atoms = structures()[MOLECULE]
    settings = make_settings(device, backend)
    ref = run(atoms, settings)
    moved = atoms.copy()
    rot = Rotation.from_euler("xyz", [0.3, -1.1, 2.0]).as_matrix()
    moved.positions = atoms.positions @ rot.T + np.array([3.1, -7.0, 0.4])
    got = run(moved, settings)
    close(got.energy, ref.energy)
    close(got.forces, ref.forces @ torch.as_tensor(rot.T, device=ref.forces.device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", [MOLECULE, CRYSTAL])
def test_permutation(name, backend, device):
    atoms = structures()[name]
    settings = make_settings(device, backend, **SHORT)
    ref = run(atoms, settings)
    perm = np.random.default_rng(0).permutation(len(atoms))
    got = run(atoms[perm], settings)
    close(got.energy, ref.energy)
    close(got.forces, ref.forces[torch.as_tensor(perm, device=ref.forces.device)])
    if ref.stress is not None:
        close(got.stress, ref.stress)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_wrapping(backend, device):
    """Fixture crystals are unwrapped (fractional coordinates -0.5..1.4)."""
    atoms = structures()[CRYSTAL]
    settings = make_settings(device, backend, **SHORT)
    ref = run(atoms, settings)
    wrapped = atoms.copy()
    wrapped.wrap()
    shifted = atoms.copy()
    shifted.positions[::2] += np.array([2, -3, 1]) @ atoms.cell.array
    for other in (wrapped, shifted):
        got = run(other, settings)
        close(got.energy, ref.energy)
        close(got.forces, ref.forces)
        close(got.stress, ref.stress)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_supercell(backend, device):
    atoms = structures()[CRYSTAL]
    settings = make_settings(device, backend, **SHORT)
    ref = run(atoms, settings)
    sc = make_supercell(atoms, np.diag([2, 1, 1]))
    got = run(sc, settings)
    close(got.energy, 2 * ref.energy)
    close(got.forces, torch.cat([ref.forces, ref.forces]))
    close(got.stress, ref.stress)
