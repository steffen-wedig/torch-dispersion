from torch import Tensor

from ..settings import BruteForceNeighbors, NeighborBackend, NvalchemiNeighbors
from .base import NeighborBuilder, NeighborMatrix, from_pairs
from .brute_force import BruteForceBuilder, brute_force_neighbors


def make_builder(backend: NeighborBackend) -> NeighborBuilder:
    if isinstance(backend, BruteForceNeighbors):
        return BruteForceBuilder(backend.pair_block_size)
    if isinstance(backend, NvalchemiNeighbors):
        from .nvalchemi import NvalchemiBuilder

        return NvalchemiBuilder(backend.method, backend.max_neighbors)
    raise TypeError(f"unknown neighbour backend {backend!r}")


def build_batched(
    backend: NeighborBackend,
    positions: Tensor,
    cells: Tensor | None,
    pbc: bool,
    cutoff: float,
    counts: list[int],
) -> list[NeighborMatrix]:
    """Neighbour lists of a batch of systems concatenated along the atom axis (``counts`` atoms
    each, cells (S, 3, 3) or None without PBC), one matrix per system with system-local indices.

    nvalchemiops builds the whole batch in one ``system_idx``-aware call; the brute-force
    backend builds each system on its own."""
    if isinstance(backend, NvalchemiNeighbors):
        from .nvalchemi import NvalchemiBuilder

        builder = NvalchemiBuilder(backend.method, backend.max_neighbors)
        return builder.batch(positions, cells, pbc, cutoff, counts)
    single = make_builder(backend)
    result, start = [], 0
    for s, n in enumerate(counts):
        cell = cells[s] if cells is not None else None
        result.append(single(positions[start : start + n], cell, pbc, cutoff))
        start += n
    return result


__all__ = [
    "NeighborBuilder",
    "NeighborMatrix",
    "BruteForceBuilder",
    "brute_force_neighbors",
    "build_batched",
    "from_pairs",
    "make_builder",
]
