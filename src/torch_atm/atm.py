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
from .kernel import Accumulators, KernelParams, TileInputs, get_step_fn, zero_accumulators
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


def _mark_system_sized(acc: Accumulators) -> None:
    """The accumulators' Np dimensions are dynamic like C6's (outputs of a compiled call are new
    tensors, so this is repeated before every call)."""
    torch._dynamo.mark_dynamic(acc.grad_pos, 0)
    torch._dynamo.mark_dynamic(acc.grad_c6, 0)
    torch._dynamo.mark_dynamic(acc.grad_c6, 1)


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
    f64 = torch.float64
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
    step = get_step_fn(need_grad, settings.compile, settings.kernel_gradient)
    if settings.compile:
        # the C6 matrix (and the accumulators) are the only system-sized kernel inputs
        torch._dynamo.mark_dynamic(c6_leaf, 0)
        torch._dynamo.mark_dynamic(c6_leaf, 1)

    # tiled layout (blocks, tiles, B, T, ...): every tile is a contiguous view with fixed strides
    n_blocks, n_tiles_max = n_padded // block, nl_atm.width // tile

    def tiled(x: Tensor) -> Tensor:
        x = x.reshape(n_blocks, block, n_tiles_max, tile, *x.shape[2:])
        return x.transpose(1, 2).contiguous()

    idx_t, mask_t, z_t, vec_t = (tiled(x) for x in (nl_atm.indices, nl_atm.mask, z_nb, vec))
    centres = torch.arange(n_padded, device=device).reshape(n_blocks, block)
    z_centres = system.numbers.reshape(n_blocks, block)
    diag = {flag: torch.tensor(flag, device=device) for flag in (True, False)}
    counts = nl_atm.count.reshape(n_blocks, block).amax(1).cpu().tolist()

    acc: Tensor | Accumulators
    acc = (
        zero_accumulators(n_padded, device)
        if need_grad
        else torch.zeros((), dtype=f64, device=device)
    )
    for kb in range(-(-n // block)):
        n_tiles = -(-int(counts[kb]) // tile)
        for p in range(n_tiles):
            for q in range(p, n_tiles):
                inputs = TileInputs(
                    centre=centres[kb],
                    idx_n=idx_t[kb, p],
                    idx_m=idx_t[kb, q],
                    mask_n=mask_t[kb, p],
                    mask_m=mask_t[kb, q],
                    z_centre=z_centres[kb],
                    z_n=z_t[kb, p],
                    z_m=z_t[kb, q],
                    r0=r0_table,
                    diagonal=diag[p == q],
                )
                if settings.compile and isinstance(acc, Accumulators):
                    _mark_system_sized(acc)
                acc = step(acc, vec_t[kb, p], vec_t[kb, q], c6_leaf, inputs, scalars)

    if isinstance(acc, Accumulators):
        energy, grad_pos, virial, grad_c6 = acc
    else:
        energy = acc

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
