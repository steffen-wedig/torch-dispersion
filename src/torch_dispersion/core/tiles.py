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
from typing import NamedTuple

import torch
from torch import Tensor

from ..neighbors import NeighborMatrix


class Accumulators(NamedTuple):
    """Running sums of the tile stage, float64. Rows of the padded system (Np)."""

    energy: Tensor  # ()
    grad_pos: Tensor  # (Np, 3) explicit dE/dR
    virial: Tensor  # (3, 3) explicit sum_r r (x) dE/dr
    grad_c6: Tensor  # (Np, Np) dE/dC6


def zero_accumulators(n_padded: int, device: torch.device) -> Accumulators:
    f64 = torch.float64
    return Accumulators(
        energy=torch.zeros((), dtype=f64, device=device),
        grad_pos=torch.zeros((n_padded, 3), dtype=f64, device=device),
        virial=torch.zeros((3, 3), dtype=f64, device=device),
        grad_c6=torch.zeros((n_padded, n_padded), dtype=f64, device=device),
    )


def zero_energy_or_accumulators(
    need_grad: bool, n_padded: int, device: torch.device
) -> Tensor | Accumulators:
    """The initial ``acc`` of a step function: a 0-d energy, or ``Accumulators``."""
    if need_grad:
        return zero_accumulators(n_padded, device)
    return torch.zeros((), dtype=torch.float64, device=device)


# system-sized (Np) dimensions of the accumulators, marked dynamic
_DYNAMIC_DIMS = {"grad_pos": (0,), "grad_c6": (0, 1), "grad_a": (0,)}


def mark_system_sized(acc) -> None:
    """The accumulators' Np dimensions are dynamic like C6's (outputs of a compiled call are new
    tensors, so this is repeated before every call)."""
    for name, dims in _DYNAMIC_DIMS.items():
        x = getattr(acc, name, None)
        if x is not None:
            for d in dims:
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


def run_tiles(
    step,
    acc,
    tables: Sequence[Tensor],
    layout: TileLayout,
    n: int,
    order: int,
    make_inputs: Callable[[TileLayout, int, tuple[int, ...]], tuple],
    scalars,
):
    """``acc = step(acc, *vectors, *tables, inputs, scalars)`` for every tile set of the blocks
    holding the n real atoms: single tiles (p,) for ``order`` 1, tile pairs (p, q), q >= p, for
    ``order`` 2, with ``vectors`` the displacement tiles of the set and ``inputs =
    make_inputs(layout, block, tiles)``. ``tables`` are the system-sized per-atom inputs (C6
    (Np, Np) first, then (Np, ...) tables); their Np dimensions are marked dynamic (a no-op
    without compilation)."""
    for i, x in enumerate(tables):
        for d in (0, 1) if i == 0 else (0,):
            torch._dynamo.mark_dynamic(x, d)
    track = isinstance(acc, tuple)
    for kb in range(-(-n // layout.block)):
        n_tiles = -(-layout.counts[kb] // layout.tile)
        for tiles in combinations_with_replacement(range(n_tiles), order):
            inputs = make_inputs(layout, kb, tiles)
            if track:
                mark_system_sized(acc)
            vectors = [layout.vectors[kb, p] for p in tiles]
            acc = step(acc, *vectors, *tables, inputs, scalars)
    return acc
