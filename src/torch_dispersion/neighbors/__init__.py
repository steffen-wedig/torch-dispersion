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


__all__ = [
    "NeighborBuilder",
    "NeighborMatrix",
    "BruteForceBuilder",
    "brute_force_neighbors",
    "from_pairs",
    "make_builder",
]
