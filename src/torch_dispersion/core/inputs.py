"""Input checks and unit conversion of the public entry points."""

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor

from ..neighbors import NeighborMatrix
from ..settings import _EvalSettings
from .coordination import CN_MIN_R2


def check_pbc(pbc: bool | Sequence[bool] | Tensor) -> bool:
    flags = [bool(x) for x in torch.as_tensor(pbc).reshape(-1).tolist()]
    if len(set(flags)) > 1:
        raise ValueError("mixed periodicity is not supported: pbc must be all True or all False")
    return flags[0]


def check_numbers(numbers: Tensor, max_number: int) -> None:
    if numbers.numel() and (int(numbers.min()) < 1 or int(numbers.max()) > max_number):
        raise ValueError(f"atomic numbers must be in 1..{max_number}")


def check_coincident(nl: NeighborMatrix, positions: Tensor, cell: Tensor | None) -> None:
    if nl.any_closer(positions, cell, CN_MIN_R2):
        raise ValueError("coincident atoms (or periodic images) found")


def to_bohr(x: Tensor, settings: _EvalSettings) -> Tensor:
    """Positions or a cell (Å) as a detached tensor in bohr, in the settings' dtype and device."""
    x = torch.as_tensor(x).detach().to(device=settings.torch_device, dtype=settings.torch_dtype)
    return x / settings.bohr_in_angstrom


class Inputs(NamedTuple):
    pbc: bool
    numbers: Tensor  # (N,) int64 on the settings device
    positions: Tensor  # (N, 3) bohr, detached
    cell: Tensor | None  # (3, 3) bohr, detached; None without PBC


def prepare(
    numbers: Tensor,
    positions: Tensor,
    cell: Tensor | None,
    pbc: bool | Sequence[bool] | Tensor,
    settings: _EvalSettings,
) -> Inputs:
    """One structure in the API's units (Å, rows of ``cell`` = lattice vectors) as ``Inputs``.
    The atomic numbers are not range-checked (each model checks against its own tables)."""
    pbc = check_pbc(pbc)
    if pbc and cell is None:
        raise ValueError("a cell is required with pbc")
    numbers = torch.as_tensor(numbers).to(device=settings.torch_device, dtype=torch.int64)
    pos = to_bohr(positions, settings)
    return Inputs(pbc, numbers, pos, to_bohr(cell, settings) if pbc and cell is not None else None)
