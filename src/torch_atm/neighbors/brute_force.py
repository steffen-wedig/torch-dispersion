"""Reference neighbour list: all pairs times all lattice translations."""

import math

import torch
from torch import Tensor

from .base import NeighborMatrix, from_pairs


def lattice_shifts(positions: Tensor, cell: Tensor, cutoff: float) -> Tensor:
    """Integer shifts (n_T, 3) covering every image within ``cutoff`` of every atom.

    Like simple-dftd3's ``get_lattice_points``, the repeat count per axis is
    ``cutoff / plane spacing``, here enlarged by the spread of the fractional coordinates so the
    list is complete for unwrapped positions too.
    """
    cell = cell.to(torch.float64)
    volume = torch.linalg.det(cell).abs()
    cross = torch.linalg.cross(cell[[1, 2, 0]], cell[[2, 0, 1]])
    spacing = volume / cross.norm(dim=-1)
    frac = positions.to(torch.float64) @ torch.linalg.inv(cell)
    spread = frac.amax(0) - frac.amin(0) if positions.shape[0] else frac.new_zeros(3)
    reps = [math.ceil(float(x)) for x in cutoff / spacing + spread]
    axes = [torch.arange(-r, r + 1, device=positions.device) for r in reps]
    return torch.cartesian_prod(*axes).reshape(-1, 3)


def brute_force_neighbors(
    positions: Tensor,
    cell: Tensor | None,
    pbc: bool,
    cutoff: float,
    pair_block_size: int = 4096,
) -> NeighborMatrix:
    n = positions.shape[0]
    device = positions.device
    positions = positions.detach()
    if pbc:
        assert cell is not None
        cell = cell.detach().to(positions.dtype)
        shifts = lattice_shifts(positions, cell, cutoff)
        translations = shifts.to(positions.dtype) @ cell
    else:
        shifts = torch.zeros((1, 3), dtype=torch.int64, device=device)
        translations = positions.new_zeros(1, 3)
    zero_shift = int(torch.nonzero((shifts == 0).all(-1))[0, 0])
    cutoff2 = cutoff * cutoff

    found_i, found_j, found_t = [], [], []
    n_pairs = n * n
    for start in range(0, n_pairs, pair_block_size):
        pair = torch.arange(start, min(start + pair_block_size, n_pairs), device=device)
        i, j = pair // n, pair % n
        vec = (positions[j] - positions[i])[:, None, :] + translations[None, :, :]
        keep = vec.square().sum(-1) <= cutoff2
        keep[i == j, zero_shift] = False
        p, t = torch.nonzero(keep, as_tuple=True)
        found_i.append(i[p])
        found_j.append(j[p])
        found_t.append(t)

    empty = torch.zeros(0, dtype=torch.int64, device=device)
    i = torch.cat(found_i) if found_i else empty
    j = torch.cat(found_j) if found_j else empty
    t = torch.cat(found_t) if found_t else empty
    return from_pairs(i, j, shifts[t], n)


class BruteForceBuilder:
    def __init__(self, pair_block_size: int = 4096):
        self.pair_block_size = pair_block_size

    def __call__(
        self, positions: Tensor, cell: Tensor | None, pbc: bool, cutoff: float
    ) -> NeighborMatrix:
        return brute_force_neighbors(positions, cell, pbc, cutoff, self.pair_block_size)
