"""D4 of a batch of systems: ``d4_dispersion_batched(...) -> D4BatchResult``.

The layout is TorchSim's: the atoms of all systems concatenated (system by system), one cell
per system and one periodicity for the whole batch. What is batched and what is per system:

- the neighbour list: one batched build at ``neighbor_list_cutoff`` (nvalchemiops'
  ``system_idx``-aware list; the brute-force backend builds per system), split into one
  matrix per system with system-local indices;
- input checks and unit conversion: once for the batch;
- everything else runs per system through ``evaluate_system``, the same code path as
  ``d4_dispersion``: CN, the EEQ solve with the system's own total charge, the (Np x Np) C6
  matrices and dE/dC6 accumulators of that system only (C6 stays block-diagonal: no batch-sized
  matrix is ever formed), the two-body and ATM tile kernels, and the chain closure.

The tile kernels' cost and memory scale with the C6 size they see (the ATM kernel's dE/dC6
scatter is one-hot over Np), so feeding them per-system blocks is what keeps a 9k-atom batch
at the cost of its systems. The tiles are already the unit of work, so the loop over systems
adds little to the loop over tiles. Each result equals a ``d4_dispersion`` call on that system.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..atm import check_pbc
from ..neighbors import build_batched
from ..settings import D4Settings
from .data import d4_reference_data
from .dispersion import check_numbers, evaluate_system, neighbor_list_cutoff, to_bohr


@dataclass
class D4BatchResult:
    energy: Tensor  # (S,) eV
    forces: Tensor | None  # (N, 3) eV/Å
    stress: Tensor | None  # (S, 3, 3) eV/Å^3, ASE convention, symmetrised; None without PBC
    virial: Tensor | None  # (S, 3, 3) eV, dE/dε; None without PBC
    charges: Tensor  # (N,) EEQ partial charges (e)


def d4_dispersion_batched(
    numbers: Tensor,
    positions: Tensor,
    cells: Tensor | None,
    pbc: bool | Sequence[bool] | Tensor,
    n_atoms: Sequence[int] | Tensor,
    settings: D4Settings,
    charges: Sequence[float] | Tensor | None = None,
    compute_forces: bool = True,
    compute_stress: bool = True,
) -> D4BatchResult:
    """DFT-D4 of S systems concatenated along the atom axis: ``numbers`` (N,), ``positions``
    (N, 3) in Å, ``cells`` (S, 3, 3) in Å with rows = lattice vectors (ignored without PBC),
    ``n_atoms`` (S,) atoms per system, ``charges`` (S,) total charges (default 0).

    ``pbc`` is one setting for the whole batch, all True or all False (a batch cannot mix
    molecules and cells). Returns float64 tensors on ``settings.device``; every system's entries
    equal ``d4_dispersion`` on that system alone.
    """
    pbc = check_pbc(pbc)
    device = settings.torch_device
    counts = [int(n) for n in torch.as_tensor(n_atoms).reshape(-1).tolist()]
    n_systems = len(counts)
    numbers = torch.as_tensor(numbers).to(device=device, dtype=torch.int64)
    if n_systems == 0 or min(counts) < 1:
        raise ValueError("every system of the batch needs at least one atom")
    if sum(counts) != numbers.shape[0]:
        raise ValueError(f"n_atoms sums to {sum(counts)}, but there are {numbers.shape[0]} atoms")
    if pbc and cells is None:
        raise ValueError("cells are required with pbc")
    total = [0.0] * n_systems
    if charges is not None:
        total = [float(q) for q in torch.as_tensor(charges).reshape(-1).tolist()]
        if len(total) != n_systems:
            raise ValueError(f"{len(total)} total charges for {n_systems} systems")
    check_numbers(numbers, d4_reference_data(device, settings.torch_dtype))
    pos = to_bohr(positions, settings)
    cells_b = cells_a = None
    if pbc and cells is not None:
        cells_b = to_bohr(cells, settings).reshape(n_systems, 3, 3)
        cells_a = torch.as_tensor(cells).reshape(n_systems, 3, 3)
    neighbors = build_batched(
        settings.neighbor_backend, pos, cells_b, pbc, neighbor_list_cutoff(settings), counts
    )

    results, start = [], 0
    for s, n in enumerate(counts):
        results.append(
            evaluate_system(
                numbers[start : start + n],
                pos[start : start + n],
                None if cells_b is None else cells_b[s],
                None if cells_a is None else cells_a[s],
                neighbors[s],
                settings,
                total[s],
                compute_forces,
                compute_stress,
            )
        )
        start += n

    def stacked(name: str, stack) -> Tensor | None:
        values = [getattr(r, name) for r in results]
        return None if values[0] is None else stack(values)

    return D4BatchResult(
        energy=torch.stack([r.energy for r in results]),
        forces=stacked("forces", torch.cat),
        stress=stacked("stress", torch.stack),
        virial=stacked("virial", torch.stack),
        charges=torch.cat([r.charges for r in results]),
    )


__all__ = ["D4BatchResult", "d4_dispersion_batched"]
