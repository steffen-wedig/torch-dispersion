"""Orchestration: ``atm_dispersion(...) -> ATMResult``.

Gradient strategy (design §5.4, ``core.chain``):

1. graph stage: positions and strain -> CN -> reference weights -> C6 (N x N), with autograd;
2. triple stage: C6 detached, eager loop over centre blocks and neighbour tile pairs; every
   kernel call returns its energy and its gradients with respect to the tile's displacement
   vectors and to C6, which are scatter-accumulated into dE/dR, the strain virial and dE/dC6;
3. ``torch.autograd.grad(C6, ..., grad_outputs=dE/dC6)`` adds the CN contribution.

Internally everything is in bohr and Hartree; the API takes and returns Å and eV.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..core.chain import close_chain, forces_and_stress, graph_fn, strained, zero_outputs
from ..core.inputs import check_coincident, check_numbers, prepare
from ..core.padding import CN_WIDTH_MULTIPLE, pad_positions, padded_size
from ..core.tiles import Accumulators, TileLayout, run_tiles, zero_energy_or_accumulators
from ..d3.data import pad_system, reference_data
from ..d3.model import cn_weights_c6
from ..neighbors import NeighborMatrix, make_builder
from ..settings import ATMSettings
from .kernel import KernelParams, get_step_fn, triple_inputs


@dataclass
class ATMResult:
    energy: Tensor  # () eV
    forces: Tensor | None  # (N, 3) eV/Å
    stress: Tensor | None  # (3, 3) eV/Å^3, ASE convention, symmetrised; None without PBC
    virial: Tensor | None  # (3, 3) eV, dE/dε (extensive); None without PBC


def _stage1(
    positions: Tensor,
    cell: Tensor | None,
    nl: NeighborMatrix,
    rcov: Tensor,
    ref_cn: Tensor,
    ref_mask: Tensor,
    species: Tensor,
    c6_species: Tensor,
    cutoff: float,
) -> tuple[Tensor, Tensor]:
    """(CN, C6) of the padded system; fixed-shape, compiles with ``fullgraph=True``."""
    vec = nl.vectors(positions, cell)
    cn, _, c6 = cn_weights_c6(
        vec, nl.mask, nl.indices, rcov, ref_cn, ref_mask, species, c6_species, cutoff
    )
    return cn, c6


def atm_dispersion(
    numbers: Tensor,
    positions: Tensor,
    cell: Tensor | None,
    pbc: bool | Sequence[bool] | Tensor,
    settings: ATMSettings,
    compute_forces: bool = True,
    compute_stress: bool = True,
) -> ATMResult:
    """ATM three-body dispersion. Positions and cell (rows = lattice vectors) in Å.

    Returns float64 tensors on ``settings.device``: energy (eV), forces (eV/Å), stress (eV/Å^3,
    ASE sign convention) and virial dE/dε (eV). Stress and virial are None without PBC.
    """
    inputs = prepare(numbers, positions, cell, pbc, settings)
    device, dtype = settings.torch_device, settings.torch_dtype
    numbers, pos, cell_b = inputs.numbers, inputs.positions, inputs.cell
    n = numbers.shape[0]
    compute_stress = compute_stress and inputs.pbc
    if settings.s9 == 0.0 or n == 0:
        return ATMResult(*zero_outputs(n, compute_forces, compute_stress, device))

    data = reference_data(device, dtype)
    check_numbers(numbers, data.max_number)

    # neighbour lists (no autograd), one build at the larger cutoff
    c3, ccn = settings.three_body_cutoff_bohr, settings.coordination_cutoff_bohr
    builder = make_builder(settings.neighbor_backend)
    nl_all = builder(pos, cell_b, inputs.pbc, max(c3, ccn))
    check_coincident(nl_all, pos, cell_b)
    nl_cn = nl_all if ccn >= c3 else nl_all.restrict(pos, cell_b, ccn)
    nl_atm = nl_all if c3 >= ccn else nl_all.restrict(pos, cell_b, c3)

    block, tile = settings.centre_block_size, settings.neighbor_tile_size
    n_padded = padded_size(n, block)
    nl_cn = nl_cn.pad(n_padded, CN_WIDTH_MULTIPLE)
    nl_atm = nl_atm.pad(n_padded, tile)
    system = pad_system(numbers, n_padded, data)

    # stage 1: CN and C6 with autograd
    need_grad = compute_forces or compute_stress
    geometry = strained(pos, cell_b, compute_forces, compute_stress)
    with torch.set_grad_enabled(need_grad):
        _, c6 = graph_fn(_stage1, settings.compile)(
            pad_positions(geometry.positions, n_padded),
            geometry.cell,
            nl_cn,
            system.rcov,
            system.ref_cn,
            system.ref_mask,
            system.species,
            system.c6_species,
            ccn,
        )

    # stage 2: blocked triples on detached C6
    pos_d = pad_positions(geometry.positions.detach(), n_padded)
    cell_d = None if geometry.cell is None else geometry.cell.detach()
    layout = TileLayout.build(nl_atm, nl_atm.vectors(pos_d, cell_d), system.numbers, block, tile)
    scalars = KernelParams(
        s9=settings.s9,
        alpha=settings.alpha,
        cutoff=c3,
        width=settings.three_body_cutoff_width_bohr,
    ).scalars(dtype, device)
    acc = run_tiles(
        get_step_fn(need_grad, settings.compile, settings.kernel_gradient),
        zero_energy_or_accumulators(need_grad, n_padded, device),
        (c6.detach(),),
        layout,
        n,
        2,
        triple_inputs(settings.rs9 * data.r0_vdw),
        scalars,
    )
    if not isinstance(acc, Accumulators):
        return ATMResult(energy=acc * settings.hartree_in_ev, forces=None, stress=None, virial=None)

    # stage 3: close the chain through CN -> C6
    grad_pos, grad_strain = close_chain(geometry, [c6], [acc.grad_c6], acc.grad_pos, acc.virial, n)
    forces, stress, virial = forces_and_stress(grad_pos, grad_strain, cell, settings)
    return ATMResult(
        energy=acc.energy * settings.hartree_in_ev, forces=forces, stress=stress, virial=virial
    )
