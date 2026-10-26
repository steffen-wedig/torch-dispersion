"""Padded neighbour matrix shared by every backend."""

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


@dataclass(frozen=True)
class NeighborMatrix:
    """Full neighbour list (both i->j and j->i) of N atoms, padded to width M.

    Excludes (i, i, T = 0), includes the self-images (i, i, T != 0). The displacement of entry
    (i, m) is ``pos[indices[i, m]] - pos[i] + shifts[i, m] @ cell``. Indices and shifts carry no
    gradient; displacements are always recomputed from the caller's positions and cell.

    Long lists (tens of thousands of entries per row at 60 bohr) are stored compactly: int32
    indices and int8 shifts (int16 beyond +-127), 8 bytes per entry; see ``compact``.
    """

    indices: Tensor  # (N, M) int32, padded with N
    shifts: Tensor  # (N, M, 3) int8 (or int16) lattice shifts, 0 in padding
    mask: Tensor  # (N, M) bool
    count: Tensor  # (N,) int64

    @property
    def n_atoms(self) -> int:
        return self.indices.shape[0]

    @property
    def width(self) -> int:
        return self.indices.shape[1]

    def vectors(self, positions: Tensor, cell: Tensor | None) -> Tensor:
        """Displacements (N, M, 3); zero in padding.

        ``positions`` has one row per matrix row. The padding index may equal that row count
        (an unpadded matrix); it then gathers an appended zero row. ``cell`` is one cell (3, 3)
        or one cell per row (N, 3, 3), as in a flat batch of systems; None without PBC.
        """
        extended = torch.cat([positions, positions.new_zeros(1, 3)])
        # index_select, not extended[indices]: each atom appears in ~M rows, and the backward of
        # advanced indexing (a deterministic index_put) serialises on repeated indices on CUDA,
        # while index_select's backward is an atomic index_add
        gathered = extended.index_select(0, self.indices.reshape(-1)).view(*self.indices.shape, 3)
        vec = gathered - positions[:, None, :]
        if cell is not None:
            vec = vec + self.shifts.to(positions.dtype) @ cell  # (N, M, 3) @ (3, 3) or (N, 3, 3)
        return torch.where(self.mask[..., None], vec, 0.0)

    def row_chunks(self) -> list[slice]:
        """Row ranges of at most ~2^22 entries: per-entry temporaries of a long list (vectors,
        distances, sort keys) are built one chunk at a time."""
        rows = max(1, (1 << 22) // max(self.width, 1))
        return [slice(r, r + rows) for r in range(0, self.n_atoms, rows)]

    def _r2(self, positions: Tensor, cell: Tensor | None, rows: slice) -> Tensor:
        """Squared lengths (rows, M) of the displacements of ``rows`` (0 in padding)."""
        part = NeighborMatrix(
            indices=self.indices[rows],
            shifts=self.shifts[rows],
            mask=self.mask[rows],
            count=self.count[rows],
        )
        # positions of the neighbours via the full array, of the centres via the row range
        extended = torch.cat([positions, positions.new_zeros(1, 3)])
        vec = extended.index_select(0, part.indices.reshape(-1)).view(*part.indices.shape, 3)
        vec = vec - positions[rows][:, None, :]
        if cell is not None:
            vec = vec + part.shifts.to(positions.dtype) @ (cell[rows] if cell.dim() == 3 else cell)
        return torch.where(part.mask, vec.square().sum(-1), 0.0)

    def within(self, positions: Tensor, cell: Tensor | None, cutoff: float) -> Tensor:
        """(N, M) bool: real entries with r^2 <= cutoff^2, computed in row chunks."""
        with torch.no_grad():
            positions = positions.detach()
            cell = None if cell is None else cell.detach()
            parts = [
                self.mask[rows] & (self._r2(positions, cell, rows) <= cutoff * cutoff)
                for rows in self.row_chunks()
            ]
        return torch.cat(parts) if parts else self.mask.clone()

    def any_closer(self, positions: Tensor, cell: Tensor | None, r2_min: float) -> bool:
        """Whether a real entry has r^2 < ``r2_min`` (row chunks, one host sync)."""
        with torch.no_grad():
            positions = positions.detach()
            cell = None if cell is None else cell.detach()
            hits = [
                (self.mask[rows] & (self._r2(positions, cell, rows) < r2_min)).any()
                for rows in self.row_chunks()
            ]
        return bool(torch.stack(hits).any()) if hits else False

    def restrict(self, positions: Tensor, cell: Tensor | None, cutoff: float) -> "NeighborMatrix":
        """Keep entries with r^2 <= cutoff^2 and compact the rows."""
        return self.select(self.within(positions, cell, cutoff))

    def select(self, keep: Tensor) -> "NeighborMatrix":
        """Compact the rows to the entries where ``keep`` is True (order preserved; row
        chunks)."""
        n = self.n_atoms
        count = keep.sum(-1)
        width = int(count.max()) if n else 0
        parts = []
        for rows in self.row_chunks():
            k = keep[rows]
            # stable sort moves the kept entries to the front of each row, in their order
            order = torch.sort((~k).to(torch.int8), dim=-1, stable=True).indices[:, :width]
            mask = torch.gather(k, 1, order)
            indices = torch.where(mask, torch.gather(self.indices[rows], 1, order), n)
            shifts = torch.gather(self.shifts[rows], 1, order[..., None].expand(-1, -1, 3))
            parts.append((indices, torch.where(mask[..., None], shifts, 0), mask))
        if not parts:
            return self
        indices, shifts, mask = (torch.cat(x) for x in zip(*parts, strict=True))
        return NeighborMatrix(indices=indices, shifts=shifts, mask=mask, count=count)

    def pad(self, n_rows: int, width_multiple: int) -> "NeighborMatrix":
        """Pad to ``n_rows`` rows (extra rows empty) and a width that is a multiple of
        ``width_multiple`` (at least one). The padding index stays ``self.n_atoms``."""
        n, m = self.indices.shape
        width = max(width_multiple, -(-m // width_multiple) * width_multiple)
        dr, dm = n_rows - n, width - m
        if dr < 0:
            raise ValueError("cannot pad to fewer rows")
        indices = torch.nn.functional.pad(self.indices, (0, dm, 0, dr), value=n)
        shifts = torch.nn.functional.pad(self.shifts, (0, 0, 0, dm, 0, dr), value=0)
        mask = torch.nn.functional.pad(self.mask, (0, dm, 0, dr), value=False)
        count = torch.nn.functional.pad(self.count, (0, dr), value=0)
        return NeighborMatrix(indices=indices, shifts=shifts, mask=mask, count=count)

    def rows(self, start: int, n: int) -> "NeighborMatrix":
        """Rows start..start+n of a flat batch matrix whose neighbours lie in the same range (one
        system), with indices local to it (padding index n)."""
        mask = self.mask[start : start + n]
        indices = torch.where(mask, self.indices[start : start + n] - start, n)
        return NeighborMatrix(
            indices=indices,
            shifts=self.shifts[start : start + n],
            mask=mask,
            count=self.count[start : start + n],
        )

    def to_pairs(self) -> tuple[Tensor, Tensor, Tensor]:
        """Flat (i, j, shift) of the real entries, i ascending."""
        i = torch.arange(self.n_atoms, device=self.indices.device)[:, None].expand_as(self.mask)
        return i[self.mask], self.indices[self.mask], self.shifts[self.mask]


def concat(matrices: list[NeighborMatrix]) -> NeighborMatrix:
    """One flat matrix of a batch from per-system matrices with system-local indices: rows are
    concatenated, indices offset to global atom indices, the width is the widest system's and
    the padding index is the total atom count."""
    counts = [m.n_atoms for m in matrices]
    n_total, width = sum(counts), max((m.width for m in matrices), default=0)
    parts, start = [], 0
    for m in matrices:
        dm = width - m.width
        mask = torch.nn.functional.pad(m.mask, (0, dm), value=False)
        indices = torch.nn.functional.pad(m.indices, (0, dm), value=0) + start
        parts.append(
            (
                torch.where(mask, indices, n_total),
                torch.nn.functional.pad(m.shifts, (0, 0, 0, dm), value=0),
                mask,
                m.count,
            )
        )
        start += m.n_atoms
    indices, shifts, mask, count = (torch.cat(x) for x in zip(*parts, strict=True))
    return NeighborMatrix(indices=indices, shifts=shifts, mask=mask, count=count)


def from_pairs(i: Tensor, j: Tensor, shifts: Tensor, n_atoms: int) -> NeighborMatrix:
    """Build a padded matrix from flat pair arrays (any order; kept stable within each i)."""
    order = torch.sort(i, stable=True).indices
    i, j, shifts = i[order], j[order], shifts[order]
    count = torch.bincount(i, minlength=n_atoms)
    width = int(count.max()) if i.numel() else 0
    offsets = torch.cumsum(count, 0) - count
    col = torch.arange(i.numel(), device=i.device) - offsets[i]
    indices = torch.full((n_atoms, width), n_atoms, dtype=torch.int32, device=i.device)
    out_shifts = torch.zeros((n_atoms, width, 3), dtype=shift_dtype(shifts), device=i.device)
    mask = torch.zeros((n_atoms, width), dtype=torch.bool, device=i.device)
    indices[i, col] = j.to(torch.int32)
    out_shifts[i, col] = shifts.to(out_shifts.dtype)
    mask[i, col] = True
    return NeighborMatrix(indices=indices, shifts=out_shifts, mask=mask, count=count)


def shift_dtype(shifts: Tensor) -> torch.dtype:
    """int8 if every lattice shift fits (one host sync, no copy of ``shifts``), else int16."""
    if shifts.numel():
        extremes = torch.stack([shifts.amax(), -shifts.amin()])
        if int(extremes.max()) > 127:
            return torch.int16
    return torch.int8


class NeighborBuilder(Protocol):
    def __call__(
        self, positions: Tensor, cell: Tensor | None, pbc: bool, cutoff: float
    ) -> NeighborMatrix:
        """Neighbours within ``cutoff`` (same length unit as positions and cell)."""
        ...
