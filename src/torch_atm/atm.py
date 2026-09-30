"""Orchestration: ``atm_dispersion(...) -> ATMResult``.

Gradient strategy (design §5.4):

1. graph stage: positions and strain -> CN -> reference weights -> C6 (N x N), with autograd;
2. triple stage: C6 detached, eager loop over centre blocks and neighbour tile pairs; every
   kernel call returns its energy and its gradients with respect to the tile's displacement
   vectors and to C6, which are scatter-accumulated into dE/dR, the strain virial and dE/dC6;
3. ``torch.autograd.grad(C6, ..., grad_outputs=dE/dC6)`` adds the CN contribution.

Internally everything is in bohr and Hartree; the API takes and returns Å and eV.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import torch
from torch import Tensor

from .c6 import pairwise_c6, reference_weights
from .coordination import CN_MIN_R2, coordination_number
from .data import MAX_REFERENCES, ReferenceData, check_numbers, reference_data
from .kernel import KernelParams, TileInputs, get_tile_fn
from .neighbors import NeighborMatrix, make_builder
from .settings import ATMSettings

CN_WIDTH_MULTIPLE = 64


@dataclass
class ATMResult:
    energy: Tensor  # () eV
    forces: Tensor | None  # (N, 3) eV/Å
    stress: Tensor | None  # (3, 3) eV/Å^3, ASE convention, symmetrised; None without PBC
    virial: Tensor | None  # (3, 3) eV, dE/dε (extensive); None without PBC


def check_pbc(pbc: bool | Sequence[bool] | Tensor) -> bool:
    flags = [bool(x) for x in torch.as_tensor(pbc).reshape(-1).tolist()]
    if len(set(flags)) > 1:
        raise ValueError("mixed periodicity is not supported: pbc must be all True or all False")
    return flags[0]


def round_up(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


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
    cn = coordination_number(vec, nl.mask, rcov, rcov[nl.indices], cutoff)
    weights = reference_weights(cn, ref_cn, ref_mask)
    return cn, pairwise_c6(weights, species, c6_species)


@cache
def _compiled_stage1():
    # N, the CN width and the species count vary between systems: one dynamic-shape graph.
    # (The triple kernel, which dominates the cost, has static shapes set by the settings.)
    return torch.compile(_stage1, fullgraph=True, dynamic=True)


def stage1_fn(compile: bool):
    return _compiled_stage1() if compile else _stage1


@dataclass
class PaddedSystem:
    """Atoms padded with dummies (Z = 0, at the origin) to ``n_padded`` >= n + 1 rows."""

    n: int
    n_padded: int
    numbers: Tensor  # (Np,)
    rcov: Tensor  # (Np,)
    ref_cn: Tensor  # (Np, R)
    ref_mask: Tensor  # (Np, R)
    species: Tensor  # (Np,)
    c6_species: Tensor  # (S, S, R, R)


def pad_system(numbers: Tensor, n_padded: int, data: ReferenceData) -> PaddedSystem:
    n = numbers.shape[0]
    z = torch.nn.functional.pad(numbers, (0, n_padded - n), value=0)
    unique, species = torch.unique(z, return_inverse=True)
    ref_mask = torch.arange(MAX_REFERENCES, device=z.device) < data.n_ref[z][:, None]
    return PaddedSystem(
        n=n,
        n_padded=n_padded,
        numbers=z,
        rcov=data.covalent_radius[z],
        ref_cn=data.reference_cn[z],
        ref_mask=ref_mask,
        species=species,
        c6_species=data.c6_ref[unique][:, unique],
    )


def pad_positions(positions: Tensor, n_padded: int) -> Tensor:
    return torch.cat([positions, positions.new_zeros(n_padded - positions.shape[0], 3)])


def _check_coincident(nl: NeighborMatrix, positions: Tensor, cell: Tensor | None) -> None:
    r2 = nl.vectors(positions, cell).square().sum(-1)
    if bool((nl.mask & (r2 < CN_MIN_R2)).any()):
        raise ValueError("coincident atoms (or periodic images) found")


def _zero_result(n, pbc, compute_forces, compute_stress, device) -> ATMResult:
    f64 = torch.float64
    return ATMResult(
        energy=torch.zeros((), dtype=f64, device=device),
        forces=torch.zeros((n, 3), dtype=f64, device=device) if compute_forces else None,
        stress=torch.zeros((3, 3), dtype=f64, device=device) if compute_stress and pbc else None,
        virial=torch.zeros((3, 3), dtype=f64, device=device) if compute_stress and pbc else None,
    )


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
    pbc = check_pbc(pbc)
    device, dtype = settings.torch_device, settings.torch_dtype
    numbers = torch.as_tensor(numbers).to(device=device, dtype=torch.int64)
    n = numbers.shape[0]
    if pbc and cell is None:
        raise ValueError("a cell is required with pbc")
    compute_stress = compute_stress and pbc
    if settings.s9 == 0.0 or n == 0:
        return _zero_result(n, pbc, compute_forces, compute_stress, device)

    data = reference_data(device, dtype)
    check_numbers(numbers, data)
    bohr, hartree = settings.bohr_in_angstrom, settings.hartree_in_ev
    pos = torch.as_tensor(positions).detach().to(device=device, dtype=dtype) / bohr
    cell_b = torch.as_tensor(cell).detach().to(device=device, dtype=dtype) / bohr if pbc else None

    # neighbour lists (no autograd), one build at the larger cutoff
    c3, ccn = settings.three_body_cutoff_bohr, settings.coordination_cutoff_bohr
    builder = make_builder(settings.neighbor_backend)
    nl_all = builder(pos, cell_b, pbc, max(c3, ccn))
    _check_coincident(nl_all, pos, cell_b)
    nl_cn = nl_all if ccn >= c3 else nl_all.restrict(pos, cell_b, ccn)
    nl_atm = nl_all if c3 >= ccn else nl_all.restrict(pos, cell_b, c3)

    block, tile = settings.centre_block_size, settings.neighbor_tile_size
    n_padded = round_up(n + 1, block)
    nl_cn = nl_cn.pad(n_padded, CN_WIDTH_MULTIPLE)
    nl_atm = nl_atm.pad(n_padded, tile)
    system = pad_system(numbers, n_padded, data)

    # stage 1: CN and C6 with autograd
    need_grad = compute_forces or compute_stress
    pos_leaf = pos.clone().requires_grad_(compute_forces)
    strain = torch.zeros((3, 3), dtype=dtype, device=device, requires_grad=compute_stress)
    deform = torch.eye(3, dtype=dtype, device=device) + strain
    pos_s = pos_leaf @ deform if pbc else pos_leaf
    cell_s = cell_b @ deform if cell_b is not None else None
    with torch.set_grad_enabled(need_grad):
        _, c6 = stage1_fn(settings.compile)(
            pad_positions(pos_s, n_padded),
            cell_s,
            nl_cn,
            system.rcov,
            system.ref_cn,
            system.ref_mask,
            system.species,
            system.c6_species,
            ccn,
        )

    # stage 2: blocked triples on detached C6
    c6_leaf = c6.detach()
    pos_d = pad_positions(pos_s.detach(), n_padded)
    vec = nl_atm.vectors(pos_d, None if cell_s is None else cell_s.detach())
    z_nb = system.numbers[nl_atm.indices]
    r0_table = settings.rs9 * data.r0_vdw
    scalars = KernelParams(
        s9=settings.s9,
        alpha=settings.alpha,
        cutoff=c3,
        width=settings.three_body_cutoff_width_bohr,
    ).scalars(dtype, device)
    tile_fn = get_tile_fn(with_grad=need_grad, compile=settings.compile)
    if settings.compile:
        # the C6 matrix is the only system-sized kernel input
        torch._dynamo.mark_dynamic(c6_leaf, 0)
        torch._dynamo.mark_dynamic(c6_leaf, 1)

    f64 = torch.float64
    energy = torch.zeros((), dtype=f64, device=device)
    grad_pos = torch.zeros((n_padded, 3), dtype=f64, device=device)
    grad_c6 = torch.zeros((n_padded, n_padded), dtype=f64, device=device)
    virial = torch.zeros((3, 3), dtype=f64, device=device)
    diag = {flag: torch.tensor(flag, device=device) for flag in (True, False)}
    counts = nl_atm.count.cpu()
    for b0 in range(0, n, block):
        rows = slice(b0, b0 + block)
        n_tiles = -(-int(counts[rows].max()) // tile)
        centre = torch.arange(b0, b0 + block, device=device)
        for p in range(n_tiles):
            cols_n = slice(p * tile, (p + 1) * tile)
            for q in range(p, n_tiles):
                cols_m = slice(q * tile, (q + 1) * tile)
                inputs = TileInputs(
                    centre=centre,
                    idx_n=nl_atm.indices[rows, cols_n].contiguous(),
                    idx_m=nl_atm.indices[rows, cols_m].contiguous(),
                    mask_n=nl_atm.mask[rows, cols_n].contiguous(),
                    mask_m=nl_atm.mask[rows, cols_m].contiguous(),
                    z_centre=system.numbers[rows],
                    z_n=z_nb[rows, cols_n].contiguous(),
                    z_m=z_nb[rows, cols_m].contiguous(),
                    r0=r0_table,
                    diagonal=diag[p == q],
                )
                # contiguous tiles: strides of a view would depend on the list width
                v_n = vec[rows, cols_n].contiguous()
                v_m = vec[rows, cols_m].contiguous()
                if not need_grad:
                    energy += tile_fn(v_n, v_m, c6_leaf, inputs, scalars)
                    continue
                e, g_n, g_m, g_c6 = tile_fn(v_n, v_m, c6_leaf, inputs, scalars)
                energy += e
                g_n, g_m = g_n.to(f64), g_m.to(f64)
                grad_pos.index_add_(0, inputs.idx_n.reshape(-1), g_n.reshape(-1, 3))
                grad_pos.index_add_(0, inputs.idx_m.reshape(-1), g_m.reshape(-1, 3))
                grad_pos.index_add_(0, centre, -(g_n.sum(1) + g_m.sum(1)))
                virial += torch.einsum("btx,bty->xy", v_n.to(f64), g_n)
                virial += torch.einsum("btx,bty->xy", v_m.to(f64), g_m)
                grad_c6 += g_c6

    forces = stress = virial_out = None
    if need_grad:
        # stage 3: close the chain through CN -> C6
        leaves = [pos_leaf] if compute_forces else []
        leaves += [strain] if compute_stress else []
        if c6.requires_grad:
            cn_grads = torch.autograd.grad(
                c6, leaves, grad_outputs=grad_c6.to(c6.dtype), allow_unused=True
            )
            cn_grads = [
                torch.zeros_like(x) if g is None else g
                for x, g in zip(leaves, cn_grads, strict=True)
            ]
        else:
            cn_grads = [torch.zeros_like(x) for x in leaves]
        if compute_forces:
            gradient = grad_pos[:n] + cn_grads.pop(0).to(f64)
            forces = -gradient * (hartree / bohr)
        if compute_stress:
            assert cell is not None
            virial_out = (virial + cn_grads.pop(0).to(f64)) * hartree
            volume = torch.linalg.det(torch.as_tensor(cell).to(device=device, dtype=f64)).abs()
            stress = 0.5 * (virial_out + virial_out.T) / volume
    return ATMResult(energy=energy * hartree, forces=forces, stress=stress, virial=virial_out)
