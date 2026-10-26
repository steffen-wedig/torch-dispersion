"""The graph stage and the chain closure of the staged gradients.

Every term is evaluated in three stages:

1. graph stage, with autograd: positions and a strain leaf -> CN (and charges) -> C6;
2. tile stage on detached C6 (``core.tiles``), returning the energy, its explicit gradients with
   respect to the positions and the strain, and dE/dC6;
3. one ``torch.autograd.grad`` of the graph-stage outputs with dE/dC6 as grad_outputs, which adds
   the implicit (CN, charge) contributions.

This module holds the parts shared by the terms: the strained leaves of stage 1, the compiled
graph-stage functions, stage 3 and the conversion to forces, virial and stress.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import torch
from torch import Tensor

from ..settings import _EvalSettings


@cache
def _compiled(fn):
    # N, the CN width and the species count vary between systems: one dynamic-shape graph.
    # (The tile kernels, which dominate the cost, have static shapes set by the settings.)
    return torch.compile(fn, fullgraph=True, dynamic=True)


def graph_fn(fn, compile: bool):
    """``fn`` compiled with dynamic shapes (cached process-wide), or ``fn`` itself."""
    return _compiled(fn) if compile else fn


@dataclass
class Strained:
    """The leaves of the graph stage and the strained geometry built from them."""

    pos_leaf: Tensor  # (n, 3) requires grad if forces are wanted
    strain: Tensor  # (3, 3), or (S, 3, 3) for a batch: zero, requires grad if stress is wanted
    positions: Tensor  # (n, 3) pos_leaf (1 + strain) with PBC, else pos_leaf
    cell: Tensor | None  # cell (1 + strain), (3, 3) or (S, 3, 3); None without PBC
    compute_forces: bool
    compute_stress: bool

    @property
    def leaves(self) -> list[Tensor]:
        return [self.pos_leaf] * self.compute_forces + [self.strain] * self.compute_stress


def strained(
    pos: Tensor,
    cell: Tensor | None,
    compute_forces: bool,
    compute_stress: bool,
    system: Tensor | None = None,
) -> Strained:
    """Leaves for dE/dR and dE/dε at ε = 0; ``compute_stress`` requires a cell. For a batch,
    ``cell`` is (S, 3, 3) and ``system`` (n,) the system of each atom: one strain per system."""
    batch = () if cell is None or cell.dim() == 2 else (cell.shape[0],)
    shape = (*batch, 3, 3)
    strain = torch.zeros(shape, dtype=pos.dtype, device=pos.device, requires_grad=compute_stress)
    deform = torch.eye(3, dtype=pos.dtype, device=pos.device) + strain
    pos_leaf = pos.clone().requires_grad_(compute_forces)
    positions = pos_leaf
    if cell is not None:
        if batch:
            assert system is not None
            positions = (pos_leaf[:, None, :] @ deform[system])[:, 0]
        else:
            positions = pos_leaf @ deform
    return Strained(
        pos_leaf=pos_leaf,
        strain=strain,
        positions=positions,
        cell=cell @ deform if cell is not None else None,
        compute_forces=compute_forces,
        compute_stress=compute_stress,
    )


def close_chain(
    geometry: Strained,
    outputs: Sequence[Tensor],
    grad_outputs: Sequence[Tensor],
    explicit_pos: Tensor,
    explicit_virial: Tensor,
    n: int,
) -> tuple[Tensor | None, Tensor | None]:
    """Stage 3: total dE/dR (n, 3) and dE/dε (3, 3) in float64 (None if not requested).

    The explicit parts (``explicit_pos`` (>= n, 3), ``explicit_virial``) plus one
    ``autograd.grad`` of the graph-stage ``outputs`` with ``grad_outputs`` (dE/doutput); outputs
    without a graph (e.g. a C6 that does not depend on the geometry) are skipped."""
    f64 = torch.float64
    leaves = geometry.leaves
    pairs = [(o, g.to(o.dtype)) for o, g in zip(outputs, grad_outputs, strict=True)]
    pairs = [(o, g) for o, g in pairs if o.requires_grad]
    chain = [torch.zeros_like(x) for x in leaves]
    if pairs:
        grads = torch.autograd.grad(
            [o for o, _ in pairs], leaves, grad_outputs=[g for _, g in pairs], allow_unused=True
        )
        chain = [
            torch.zeros_like(x) if g is None else g for x, g in zip(leaves, grads, strict=True)
        ]
    grad_pos = explicit_pos[:n] + chain.pop(0).to(f64) if geometry.compute_forces else None
    grad_strain = explicit_virial + chain.pop(0).to(f64) if geometry.compute_stress else None
    return grad_pos, grad_strain


def forces_and_stress(
    grad_pos: Tensor | None,
    grad_strain: Tensor | None,
    cell: Tensor | None,
    settings: _EvalSettings,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """(forces eV/Å, stress eV/Å^3 (ASE sign, symmetrised), virial dE/dε eV) from dE/dR and
    dE/dε in atomic units; ``cell`` in Å (for the volume), (3, 3) or (S, 3, 3) with dE/dε."""
    hartree, bohr = settings.hartree_in_ev, settings.bohr_in_angstrom
    forces = stress = virial = None
    if grad_pos is not None:
        forces = -grad_pos * (hartree / bohr)
    if grad_strain is not None:
        assert cell is not None
        virial = grad_strain * hartree
        f64 = torch.float64
        volume = torch.linalg.det(torch.as_tensor(cell).to(device=virial.device, dtype=f64)).abs()
        stress = 0.5 * (virial + virial.mT) / volume[..., None, None]
    return forces, stress, virial


def zero_outputs(
    n: int, compute_forces: bool, compute_stress: bool, device: torch.device
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """(energy, forces, stress, virial) of a term that vanishes (no atoms, or a zero scale)."""
    f64 = torch.float64
    zeros = torch.zeros((3, 3), dtype=f64, device=device)
    return (
        torch.zeros((), dtype=f64, device=device),
        torch.zeros((n, 3), dtype=f64, device=device) if compute_forces else None,
        zeros if compute_stress else None,
        zeros.clone() if compute_stress else None,
    )
