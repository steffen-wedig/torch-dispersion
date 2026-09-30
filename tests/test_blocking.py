"""T10: results independent of centre_block_size, neighbor_tile_size and pair_block_size."""

import pytest
import torch
from conftest import BACKENDS, DEVICES, make_settings, run, structures

from torch_atm.settings import BruteForceNeighbors

SHORT = {"three_body_cutoff_bohr": 20.0, "coordination_cutoff_bohr": 20.0}


def close(a, b):
    torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", ["dmso_crystal", "benzamide_monomer_C7H7NO"])
def test_block_sizes(name, backend, device):
    atoms = structures()[name]
    ref = run(atoms, make_settings(device, backend, **SHORT))
    for block in (1, 7, 16, len(atoms)):
        for tile in (32, 64, 256):
            got = run(
                atoms,
                make_settings(
                    device, backend, centre_block_size=block, neighbor_tile_size=tile, **SHORT
                ),
            )
            close(got.energy, ref.energy)
            close(got.forces, ref.forces)
            if ref.stress is not None:
                close(got.stress, ref.stress)


@pytest.mark.parametrize("device", DEVICES)
def test_pair_block_size(device):
    atoms = structures()["dmso_crystal"]
    ref = run(atoms, make_settings(device, **SHORT))
    for size in (1, 37, 1_000_000):
        backend = BruteForceNeighbors(pair_block_size=size)
        got = run(
            atoms, make_settings(device, **SHORT).model_copy(update={"neighbor_backend": backend})
        )
        # not bitwise on GPU: index_add_ accumulates with atomics
        close(got.energy, ref.energy)
        close(got.forces, ref.forces)
