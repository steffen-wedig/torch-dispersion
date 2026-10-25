"""Padding of a system to a multiple of the centre block size.

Padded rows are dummy atoms (Z = 0, at the origin, without references or neighbours). There is
always at least one, so index n is a valid dummy target for the padded neighbour matrices.
"""

import torch
from torch import Tensor

# neighbour-axis padding bucket of the CN neighbour matrices (graph stage)
CN_WIDTH_MULTIPLE = 64


def round_up(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


def padded_size(n: int, block: int) -> int:
    """Rows of a system of n atoms: n + 1 (at least one dummy), rounded up to the block size."""
    return round_up(n + 1, block)


def pad_numbers(numbers: Tensor, n_padded: int) -> tuple[Tensor, Tensor, Tensor]:
    """(padded atomic numbers (Np,), unique elements (S,) including 0, species index (Np,))."""
    z = torch.nn.functional.pad(numbers, (0, n_padded - numbers.shape[0]), value=0)
    unique, species = torch.unique(z, return_inverse=True)
    return z, unique, species


def pad_positions(positions: Tensor, n_padded: int) -> Tensor:
    return torch.cat([positions, positions.new_zeros(n_padded - positions.shape[0], 3)])
