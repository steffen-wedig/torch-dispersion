"""T7: the nvalchemiops neighbour set equals brute force exactly."""

import numpy as np
import pytest
import torch
from conftest import BOHR, DEVICES, names, random_cell, structures

from torch_dispersion.neighbors import NeighborMatrix, brute_force_neighbors


def nvalchemi(*args, **kwargs):
    from torch_dispersion.neighbors.nvalchemi import NvalchemiBuilder

    return NvalchemiBuilder(**kwargs)(*args)


def pair_set(nl: NeighborMatrix) -> set:
    i, j, s = nl.to_pairs()
    return set(zip(i.tolist(), j.tolist(), map(tuple, s.tolist()), strict=True))


def inputs(atoms, device):
    pbc = bool(atoms.pbc.all())
    pos = torch.tensor(atoms.positions, device=device) / BOHR
    cell = torch.tensor(atoms.cell.array, device=device) / BOHR if pbc else None
    return pos, cell, pbc


def unwrapped_small_cell():
    atoms = random_cell(3, seed=2, cell_lengths=(3.0, 3.3, 3.1))
    atoms.positions += np.array([[0, 0, 0], [3.0, -6.6, 0.0], [-9.0, 0.0, 12.4]])
    return atoms


CASES = {
    **{n: (lambda n=n: structures()[n]) for n in names("crystal") + names("monomer")},
    "random_cell": lambda: random_cell(5, seed=4),
    "unwrapped_small_cell": unwrapped_small_cell,
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cutoff", [25.0, 40.0])
@pytest.mark.parametrize("method", [None, "naive", "cell_list"])
@pytest.mark.parametrize("case", list(CASES))
def test_same_neighbor_set(case, method, cutoff, device):
    pos, cell, pbc = inputs(CASES[case](), device)
    ref = brute_force_neighbors(pos, cell, pbc, cutoff)
    got = nvalchemi(pos, cell, pbc, cutoff, method=method)
    assert pair_set(got) == pair_set(ref)
    # every displacement recomputed from the caller's (unwrapped) positions is inside the cutoff
    r2 = got.vectors(pos, cell).square().sum(-1)
    assert bool((r2[got.mask] <= cutoff**2 * (1 + 1e-12)).all())
    if pbc:
        i, j, _ = got.to_pairs()
        assert bool((i == j).any()), "self-images must be present in small cells"
        # several images of the same pair
        assert len({(a, b) for a, b in zip(i.tolist(), j.tolist(), strict=True)}) < i.numel()


@pytest.mark.parametrize("device", DEVICES)
def test_overflow_raises(device):
    pos, cell, pbc = inputs(structures()["dmso_crystal"], device)
    with pytest.raises(RuntimeError, match="overflow"):
        nvalchemi(pos, cell, pbc, 40.0, max_neighbors=100)
