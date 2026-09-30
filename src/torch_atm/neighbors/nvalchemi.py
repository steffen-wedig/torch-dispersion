"""``nvalchemiops`` (warp) neighbour-list backend.

The library returns (neighbor_matrix, num_neighbors[, neighbor_matrix_shifts]) as int32, padded
with ``fill_value``. ``num_neighbors`` is the true count even when it exceeds the allocated width,
which is how overflow is detected; overflow is always an error, never a silent truncation.
"""

import math

import torch
from torch import Tensor

from .base import NeighborMatrix

HEADROOM = 1.3
MIN_EXTRA = 32


def estimate_max_neighbors(n: int, cell: Tensor | None, pbc: bool, cutoff: float) -> int:
    if not pbc:
        return max(n, 1)
    assert cell is not None
    density = n / float(torch.linalg.det(cell.to(torch.float64)).abs())
    return math.ceil(HEADROOM * density * 4.0 / 3.0 * math.pi * cutoff**3) + MIN_EXTRA


class NvalchemiBuilder:
    def __init__(self, method: str | None = None, max_neighbors: int | None = None):
        from nvalchemiops.torch.neighbors import neighbor_list

        self._neighbor_list = neighbor_list
        self.method = method
        self.max_neighbors = max_neighbors

    def __call__(
        self, positions: Tensor, cell: Tensor | None, pbc: bool, cutoff: float
    ) -> NeighborMatrix:
        n = positions.shape[0]
        positions = positions.detach()
        max_neighbors = self.max_neighbors or estimate_max_neighbors(n, cell, pbc, cutoff)
        kwargs: dict = {"max_neighbors": max_neighbors, "fill_value": n}
        if self.method is not None:
            kwargs["method"] = self.method
        if pbc:
            assert cell is not None
            kwargs["cell"] = cell.detach().to(positions.dtype)[None]
            kwargs["pbc"] = torch.ones((1, 3), dtype=torch.bool, device=positions.device)
        out = self._neighbor_list(positions, cutoff, **kwargs)
        matrix, num = out[0], out[1]
        count = num.to(torch.int64)
        if n and int(count.max()) > matrix.shape[1]:
            raise RuntimeError(
                f"nvalchemiops neighbour overflow: {int(count.max())} neighbours for "
                f"max_neighbors={matrix.shape[1]}; set NvalchemiNeighbors.max_neighbors higher"
            )
        width = int(count.max()) if n else 0
        mask = torch.arange(width, device=matrix.device)[None, :] < count[:, None]
        indices = torch.where(mask, matrix[:, :width].to(torch.int64), n)
        if pbc:
            shifts = out[2][:, :width].to(torch.int64)
            shifts = torch.where(mask[..., None], shifts, 0)
        else:
            shifts = torch.zeros((n, width, 3), dtype=torch.int64, device=matrix.device)
        return NeighborMatrix(indices=indices, shifts=shifts, mask=mask, count=count)
