"""The tile stage: fixed-shape kernel calls over centre blocks and neighbour tiles.

A padded neighbour matrix (Np, M) is reshaped into tiles of ``block`` centres times ``tile``
neighbours, laid out (blocks, tiles, B, T, ...) so that every tile is a contiguous view with
fixed strides. The pair kernel visits each tile p of a block, the triple kernel each tile pair
p <= q. Kernel calls see fixed (B, T) shapes; the only system-sized inputs are the per-atom
tables (C6 (Np, Np) first) and the accumulators, whose Np dimensions are marked dynamic, so one
compiled graph serves every system.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Any, NamedTuple

import torch
from torch import Tensor

from ..neighbors import NeighborMatrix


class Accumulators(NamedTuple):
    """Running sums of the tile stage, float64. Rows of the padded system (Np).

    For one system: energy (), virial (3, 3), dE/dC6 (Np, Np). For a flat batch of S systems
    (the two-body kernel): energy (S,), virial (S, 3, 3) and dE/dC6 over the flattened
    per-system C6 blocks (S * n_max^2,)."""

    energy: Tensor  # () or (S,)
    grad_pos: Tensor  # (Np, 3) explicit dE/dR
    virial: Tensor  # (3, 3) or (S, 3, 3) explicit sum_r r (x) dE/dr
    grad_c6: Tensor  # (Np, Np) or (S * n_max^2,) dE/dC6


def zero_accumulators(
    n_padded: int,
    device: torch.device,
    n_systems: int | None = None,
    c6_numel: int | None = None,
) -> Accumulators:
    """Zero accumulators of one system, or of a flat batch of ``n_systems`` systems whose C6
    blocks have ``c6_numel`` entries in all."""
    f64 = torch.float64
    batch = () if n_systems is None else (n_systems,)
    c6 = (n_padded, n_padded) if c6_numel is None else (c6_numel,)
    return Accumulators(
        energy=torch.zeros(batch, dtype=f64, device=device),
        grad_pos=torch.zeros((n_padded, 3), dtype=f64, device=device),
        virial=torch.zeros((*batch, 3, 3), dtype=f64, device=device),
        grad_c6=torch.zeros(c6, dtype=f64, device=device),
    )


def zero_energy_or_accumulators(
    need_grad: bool,
    n_padded: int,
    device: torch.device,
    n_systems: int | None = None,
    c6_numel: int | None = None,
) -> Tensor | Accumulators:
    """The initial ``acc`` of a step function: the energy alone ((), or (S,) for a batch), or
    ``Accumulators``."""
    if need_grad:
        return zero_accumulators(n_padded, device, n_systems, c6_numel)
    batch = () if n_systems is None else (n_systems,)
    return torch.zeros(batch, dtype=torch.float64, device=device)


def mark_system_sized(acc) -> None:
    """The accumulators' system-sized dimensions (Np, S, the C6 entries) are dynamic like C6's
    (outputs of a compiled call are new tensors, so this is repeated before every call)."""
    if isinstance(acc, Tensor):
        dims = {"energy": acc}
    else:
        dims = {name: getattr(acc, name) for name in acc._fields}
    for name, x in dims.items():
        if name == "virial":
            marked = range(x.dim() - 2)
        elif name == "grad_c6":
            marked = range(x.dim())
        else:
            marked = range(min(x.dim(), 1))
        for d in marked:
            torch._dynamo.mark_dynamic(x, d)


@dataclass
class TileLayout:
    """A padded neighbour matrix and its displacement vectors in tiled layout."""

    block: int
    tile: int
    centres: Tensor  # (blocks, B) atom index of the centres
    z_centres: Tensor  # (blocks, B) atomic numbers of the centres (0 for dummies)
    indices: Tensor  # (blocks, tiles, B, T) neighbour atom index
    mask: Tensor  # (blocks, tiles, B, T) bool
    z: Tensor  # (blocks, tiles, B, T) neighbour atomic numbers
    vectors: Tensor  # (blocks, tiles, B, T, 3) bohr
    counts: list[int]  # per block, the longest neighbour list

    @classmethod
    def build(
        cls, nl: NeighborMatrix, vectors: Tensor, numbers: Tensor, block: int, tile: int
    ) -> "TileLayout":
        """``nl`` padded to (Np, multiple of ``tile``), ``vectors`` its (Np, M, 3) displacement
        vectors, ``numbers`` the padded atomic numbers (Np,)."""
        n_padded = numbers.shape[0]
        n_blocks, n_tiles_max = n_padded // block, nl.width // tile

        def tiled(x: Tensor) -> Tensor:
            x = x.reshape(n_blocks, block, n_tiles_max, tile, *x.shape[2:])
            return x.transpose(1, 2).contiguous()

        tables = (nl.indices, nl.mask, numbers[nl.indices], vectors)
        indices, mask, z, vec = (tiled(x) for x in tables)
        counts = nl.count.reshape(n_blocks, block).amax(1).cpu().tolist()
        return cls(
            block=block,
            tile=tile,
            centres=torch.arange(n_padded, device=vectors.device).reshape(n_blocks, block),
            z_centres=numbers.reshape(n_blocks, block),
            indices=indices,
            mask=mask,
            z=z,
            vectors=vec,
            counts=[int(c) for c in counts],
        )


@dataclass
class SliceLayout:
    """Tiles cut from an untiled neighbour matrix only when they are used, for long lists (the
    60 bohr two-body list of a whole batch): no tiled copies and no displacement vectors of the
    whole list; the kernel builds each tile's vectors from positions, indices and shifts. Each
    tile is a (B, T) copy, padded where the rows or the width run out."""

    block: int
    tile: int
    nl: NeighborMatrix  # (N, M), any width; rows beyond N are empty
    centres: Tensor  # (blocks, B) atom index of the centres
    counts: list[int]  # per block, the longest neighbour list
    vectors = None  # the kernel computes its own

    @classmethod
    def build(cls, nl: NeighborMatrix, n_padded: int, block: int, tile: int) -> "SliceLayout":
        """``nl`` with N rows (any width) for a system padded to ``n_padded`` rows."""
        count = torch.nn.functional.pad(nl.count, (0, n_padded - nl.n_atoms))
        counts = count.reshape(-1, block).amax(1).cpu().tolist()
        centres = torch.arange(n_padded, device=nl.indices.device).reshape(-1, block)
        return cls(block, tile, nl, centres, [int(c) for c in counts])

    def entries(self, x: Tensor, kb: int, p: int, fill: float | bool) -> Tensor:
        """Tile p of centre block kb of a per-entry table (N, M, ...) as a (B, T, ...) copy."""
        r0, c0 = kb * self.block, p * self.tile
        part = x[r0 : r0 + self.block, c0 : c0 + self.tile]
        dr, dc = self.block - part.shape[0], self.tile - part.shape[1]
        if dr or dc:
            pad = (0, 0) * (x.dim() - 2) + (0, dc, 0, dr)
            return torch.nn.functional.pad(part, pad, value=fill)
        return part.contiguous()


def run_tiles(
    step,
    acc,
    tables: Sequence[Tensor],
    layout,
    n: int,
    order: int,
    make_inputs: Callable[[Any, int, tuple[int, ...]], tuple],
    scalars,
):
    """``acc = step(acc, *vectors, *tables, inputs, scalars)`` for every tile set of the blocks
    holding the n real atoms: single tiles (p,) for ``order`` 1, tile pairs (p, q), q >= p, for
    ``order`` 2, with ``vectors`` the displacement tiles of the set (none for a
    ``SliceLayout``, whose kernel builds them) and ``inputs = make_inputs(layout, block,
    tiles)``. ``layout`` is a ``TileLayout`` or a ``SliceLayout``. ``tables`` are the
    system-sized per-atom inputs (C6 first, (Np, Np) or flat, then (Np, ...) tables); their
    system-sized dimensions are marked dynamic (a no-op without compilation)."""
    for i, x in enumerate(tables):
        for d in range(x.dim()) if i == 0 else (0,):
            torch._dynamo.mark_dynamic(x, d)
    track = isinstance(acc, tuple) or acc.dim() > 0
    for kb in range(-(-n // layout.block)):
        n_tiles = -(-layout.counts[kb] // layout.tile)
        for tiles in combinations_with_replacement(range(n_tiles), order):
            inputs = make_inputs(layout, kb, tiles)
            if track:
                mark_system_sized(acc)
            vectors = [] if layout.vectors is None else [layout.vectors[kb, p] for p in tiles]
            acc = step(acc, *vectors, *tables, inputs, scalars)
    return acc
