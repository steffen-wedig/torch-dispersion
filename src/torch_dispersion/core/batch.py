"""Layout of a batch of systems concatenated along the atom axis.

Atoms are flat (N rows, system by system). Quantities that couple the atoms of one system densely
(the EEQ Coulomb matrix, C6) live in a padded layout (S, n_max, ...), where row k of system s is
its k-th atom and the remaining rows are empty.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Batch:
    counts: tuple[int, ...]  # atoms per system
    n_max: int  # atoms of the largest system
    system: Tensor  # (N,) int64 system of each atom
    local: Tensor  # (N,) int64 index of each atom within its system

    @classmethod
    def from_counts(cls, counts: list[int] | tuple[int, ...], device: torch.device) -> "Batch":
        sizes = torch.tensor(list(counts), device=device)
        system = torch.repeat_interleave(torch.arange(len(counts), device=device), sizes)
        offsets = torch.cumsum(sizes, 0) - sizes
        local = torch.arange(system.numel(), device=device) - offsets[system]
        return cls(tuple(counts), max(counts), system, local)

    @property
    def n_systems(self) -> int:
        return len(self.counts)

    @property
    def n_atoms(self) -> int:
        return self.system.numel()

    @property
    def index(self) -> Tensor:
        """(N,) row of each atom in the flattened padded layout (S * n_max)."""
        return self.system * self.n_max + self.local

    @property
    def mask(self) -> Tensor:
        """(S, n_max) True for real atoms."""
        sizes = torch.tensor(self.counts, device=self.system.device)
        return torch.arange(self.n_max, device=self.system.device) < sizes[:, None]

    def pad(self, x: Tensor, fill: float = 0.0) -> Tensor:
        """(N, ...) -> (S, n_max, ...), ``fill`` in the empty rows; differentiable."""
        return pad_to_systems(x, self.index, self.n_systems, self.n_max, fill)


def pad_to_systems(x: Tensor, index: Tensor, n_systems: int, n_max: int, fill: float = 0.0):
    """``Batch.pad`` from plain tensors (for compiled functions): (N, ...) -> (S, n_max, ...)."""
    out = x.new_full((n_systems * n_max, *x.shape[1:]), fill)
    return out.index_copy(0, index, x).view(n_systems, n_max, *x.shape[1:])
