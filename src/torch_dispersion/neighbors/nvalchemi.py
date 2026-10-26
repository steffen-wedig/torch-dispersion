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
        return _to_matrix(matrix, count, out[2] if pbc else None, 0, width)

    def batch(
        self, positions: Tensor, cells: Tensor | None, pbc: bool, cutoff: float, counts: list[int]
    ) -> NeighborMatrix:
        """The flat list of a batch of systems concatenated along the atom axis (``counts``
        atoms each, cells (S, 3, 3) or None without PBC) from one batched (``system_idx``-aware)
        library call: global indices, padding index = total atom count."""
        device = positions.device
        positions = positions.detach()
        n_total, n_systems = positions.shape[0], len(counts)
        sizes = torch.tensor(counts, device=device)
        ptr = torch.nn.functional.pad(torch.cumsum(sizes, 0), (1, 0)).to(torch.int32)
        system = torch.repeat_interleave(torch.arange(n_systems, device=device), sizes)
        max_neighbors = self.max_neighbors or _estimate_batch(counts, cells, pbc, cutoff)
        kwargs: dict = {
            "max_neighbors": max_neighbors,
            "fill_value": n_total,
            "batch_idx": system.to(torch.int32),
            "batch_ptr": ptr,
        }
        if self.method is not None:
            kwargs["method"] = self.method
        if pbc:
            assert cells is not None
            kwargs["cell"] = cells.detach().to(positions.dtype)
            kwargs["pbc"] = torch.ones((n_systems, 3), dtype=torch.bool, device=device)
        out = self._neighbor_list(positions, cutoff, **kwargs)
        matrix, count = out[0], out[1].to(torch.int64)
        width = int(count.max()) if n_total else 0  # one host sync: overflow check and width
        if width > matrix.shape[1]:
            raise RuntimeError(
                f"nvalchemiops neighbour overflow: {width} neighbours for "
                f"max_neighbors={matrix.shape[1]}; set NvalchemiNeighbors.max_neighbors higher"
            )
        return _to_matrix(matrix, count, out[2] if pbc else None, 0, width)


def _estimate_batch(counts: list[int], cells: Tensor | None, pbc: bool, cutoff: float) -> int:
    """``estimate_max_neighbors`` of the densest system of a batch (one host sync)."""
    if not pbc:
        return max(max(counts), 1)
    assert cells is not None
    volumes = torch.linalg.det(cells.to(torch.float64)).abs().tolist()
    density = max(n / v for n, v in zip(counts, volumes, strict=True))
    return math.ceil(HEADROOM * density * 4.0 / 3.0 * math.pi * cutoff**3) + MIN_EXTRA


def _to_matrix(
    matrix: Tensor, count: Tensor, shifts: Tensor | None, offset: int, width: int
) -> NeighborMatrix:
    """Library rows (indices offset by ``offset``) as a NeighborMatrix of ``width`` columns,
    padded with the local atom count."""
    n = matrix.shape[0]
    mask = torch.arange(width, device=matrix.device)[None, :] < count[:, None]
    indices = torch.where(mask, matrix[:, :width].to(torch.int64) - offset, n)
    if shifts is not None:
        shifts = torch.where(mask[..., None], shifts[:, :width].to(torch.int64), 0)
    else:
        shifts = torch.zeros((n, width, 3), dtype=torch.int64, device=matrix.device)
    return NeighborMatrix(indices=indices, shifts=shifts, mask=mask, count=count)
