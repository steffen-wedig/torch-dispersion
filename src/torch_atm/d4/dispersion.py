"""Orchestration: ``d4_dispersion(...) -> D4Result`` (dftd4 4.3.0 ``get_dispersion``).

Gradient strategy, as for the D3 ATM term (``torch_atm.atm``):

1. graph stage (autograd): positions and strain -> D4 CN, EEQ CN -> EEQ charges (Coulomb matrix,
   bordered solve) -> charge-scaled reference weights -> C6(q) (two-body) and C6(q = 0) (ATM);
2. blocked stage on detached C6: the two-body pair tiles and the ATM triple tiles each return
   their energy and gradients with respect to the displacement vectors and to their C6 matrix,
   accumulated into dE/dR, the strain virial, dE/dC6(q) and dE/dC6(0);
3. one ``torch.autograd.grad`` through both C6 matrices adds the CN and charge contributions.

Internally everything is in bohr and Hartree; the API takes and returns Å and eV. The EEQ stage
always runs in float64 (it is O(N^2) and its Ewald sums cancel strongly).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import torch
from torch import Tensor

from ..atm import (
    CN_WIDTH_MULTIPLE,
    _check_coincident,
    _mark_system_sized,
    check_pbc,
    pad_positions,
    round_up,
    run_tiles,
)
from ..c6 import pairwise_c6
from ..kernel import Accumulators, KernelParams, get_step_fn, zero_accumulators
from ..neighbors import NeighborMatrix, make_builder
from ..settings import D4Settings
from . import eeq
from .data import D4ReferenceData, d4_reference_data, reference_c6
from .model import d4_coordination_number, reference_weights
from .two_body import PairTileInputs, get_pair_step_fn, pair_scalars


@dataclass
class D4Result:
    energy: Tensor  # () eV, two-body + three-body
    forces: Tensor | None  # (N, 3) eV/Å
    stress: Tensor | None  # (3, 3) eV/Å^3, ASE convention, symmetrised; None without PBC
    virial: Tensor | None  # (3, 3) eV, dE/dε (extensive); None without PBC
    two_body_energy: Tensor  # () eV
    three_body_energy: Tensor  # () eV
    charges: Tensor  # (N,) EEQ partial charges (e)


@dataclass
class _System:
    """Atoms padded with dummies (Z = 0, at the origin, no references) to ``n_padded`` rows."""

    n: int
    numbers: Tensor  # (Np,)
    species: Tensor  # (Np,) index into the unique elements
    c6_species: Tensor  # (S, S, R, R) reference C6
    rcov: Tensor  # (Np,)
    en: Tensor  # (Np,)
    reference_cn: Tensor  # (Np, R)
    reference_q: Tensor  # (Np, R)
    n_gaussians: Tensor  # (Np, R)
    zeff: Tensor  # (Np,)
    hardness: Tensor  # (Np,)


def _pad_system(numbers: Tensor, n_padded: int, data: D4ReferenceData) -> _System:
    z = torch.nn.functional.pad(numbers, (0, n_padded - numbers.shape[0]), value=0)
    unique, species = torch.unique(z, return_inverse=True)
    alpha = data.alpha_iw[unique]
    return _System(
        n=numbers.shape[0],
        numbers=z,
        species=species,
        c6_species=reference_c6(alpha, alpha),
        rcov=data.covalent_radius[z],
        en=data.electronegativity[z],
        reference_cn=data.reference_cn[z],
        reference_q=data.reference_q[z],
        n_gaussians=data.n_gaussians[z],
        zeff=data.zeff[z],
        hardness=data.hardness[z],
    )


def _coordination_numbers(
    vec_cn: Tensor,
    mask_cn: Tensor,
    idx_cn: Tensor,
    vec_eeq: Tensor,
    mask_eeq: Tensor,
    idx_eeq: Tensor,
    rcov: Tensor,
    en: Tensor,
    cutoff: float,
) -> tuple[Tensor, Tensor]:
    """(D4 CN, EEQ CN) of the padded system; fixed-shape, compiles with ``fullgraph=True``."""
    cn = d4_coordination_number(vec_cn, mask_cn, rcov, rcov[idx_cn], en, en[idx_cn], cutoff)
    return cn, eeq.eeq_coordination_number(vec_eeq, mask_eeq, rcov, rcov[idx_eeq])


def _c6_pair(
    cn: Tensor,
    q: Tensor,
    reference_cn: Tensor,
    reference_q: Tensor,
    n_gaussians: Tensor,
    zeff: Tensor,
    hardness: Tensor,
    species: Tensor,
    c6_species: Tensor,
) -> tuple[Tensor, Tensor]:
    """C6 with the EEQ charges (two-body) and with q = 0 (ATM)."""
    args = (reference_cn, reference_q, n_gaussians, zeff, hardness)
    w_q = reference_weights(cn, q, *args)
    w_0 = reference_weights(cn, torch.zeros_like(q), *args)
    return pairwise_c6(w_q, species, c6_species), pairwise_c6(w_0, species, c6_species)


@cache
def _compiled(fn):
    return torch.compile(fn, fullgraph=True, dynamic=True)


def _maybe_compiled(fn, compile: bool):
    return _compiled(fn) if compile else fn


def _charges(
    numbers: Tensor,
    pos: Tensor,
    cell: Tensor | None,
    cell_ref: Tensor | None,
    cn_eeq: Tensor,
    data: D4ReferenceData,
    total_charge: float,
) -> Tensor:
    """EEQ charges of the n real atoms (float64). ``cell_ref`` (unstrained, detached) fixes the
    Wigner-Seitz images and the Ewald splitting."""
    f64 = torch.float64
    pos, cn_eeq = pos.to(f64), cn_eeq.to(f64)
    rad, eta = data.eeq_rad.to(f64)[numbers], data.eeq_eta.to(f64)[numbers]
    x = eeq.electronegativity_vector(
        cn_eeq, data.eeq_chi.to(f64)[numbers], data.eeq_kcnchi.to(f64)[numbers]
    )
    if cell is None:
        a = eeq.coulomb_matrix_molecular(pos, rad, eta)
    else:
        assert cell_ref is not None
        alpha = eeq.ewald_alpha(cell_ref)
        pairs = eeq.wigner_seitz_images(pos.detach(), cell_ref.to(f64))
        a = eeq.coulomb_matrix_periodic(pos, cell.to(f64), rad, eta, alpha, pairs)
    return eeq.solve_charges(a, x, total_charge)


def run_pair_tiles(
    step,
    acc,
    c6: Tensor,
    nl: NeighborMatrix,
    vec: Tensor,
    numbers: Tensor,
    n: int,
    block: int,
    tile: int,
    rr: Tensor,
    r0: Tensor,
    scalars,
):
    """Loop over centre blocks and neighbour tiles: ``acc = step(acc, v, c6, inputs, scalars)``."""
    n_padded = numbers.shape[0]
    for d in (0, 1):
        torch._dynamo.mark_dynamic(c6, d)
    n_blocks, n_tiles_max = n_padded // block, nl.width // tile

    def tiled(x: Tensor) -> Tensor:
        x = x.reshape(n_blocks, block, n_tiles_max, tile, *x.shape[2:])
        return x.transpose(1, 2).contiguous()

    idx_t, mask_t, z_t, vec_t = (tiled(x) for x in (nl.indices, nl.mask, numbers[nl.indices], vec))
    centres = torch.arange(n_padded, device=vec.device).reshape(n_blocks, block)
    z_centres = numbers.reshape(n_blocks, block)
    counts = nl.count.reshape(n_blocks, block).amax(1).cpu().tolist()
    track = isinstance(acc, tuple)
    for kb in range(-(-n // block)):
        for p in range(-(-int(counts[kb]) // tile)):
            inputs = PairTileInputs(
                centre=centres[kb],
                idx=idx_t[kb, p],
                mask=mask_t[kb, p],
                z_centre=z_centres[kb],
                z=z_t[kb, p],
                rr=rr,
                r0=r0,
            )
            if track:
                _mark_system_sized(acc)
            acc = step(acc, vec_t[kb, p], c6, inputs, scalars)
    return acc


def d4_dispersion(
    numbers: Tensor,
    positions: Tensor,
    cell: Tensor | None,
    pbc: bool | Sequence[bool] | Tensor,
    settings: D4Settings,
    charge: float = 0.0,
    compute_forces: bool = True,
    compute_stress: bool = True,
) -> D4Result:
    """DFT-D4 dispersion (two-body rational damping + ATM). Positions and cell (rows = lattice
    vectors) in Å; ``charge`` is the total charge distributed by the EEQ model.

    Returns float64 tensors on ``settings.device``: energy (eV), forces (eV/Å), stress (eV/Å^3,
    ASE sign convention), virial dE/dε (eV); stress and virial are None without PBC.
    """
    pbc = check_pbc(pbc)
    device, dtype = settings.torch_device, settings.torch_dtype
    f64 = torch.float64
    numbers = torch.as_tensor(numbers).to(device=device, dtype=torch.int64)
    n = numbers.shape[0]
    if pbc and cell is None:
        raise ValueError("a cell is required with pbc")
    compute_stress = compute_stress and pbc
    if n == 0:
        z = torch.zeros((), dtype=f64, device=device)
        empty = torch.zeros((0,), dtype=f64, device=device)
        return D4Result(
            energy=z,
            forces=torch.zeros((0, 3), dtype=f64, device=device) if compute_forces else None,
            stress=torch.zeros((3, 3), dtype=f64, device=device) if compute_stress else None,
            virial=torch.zeros((3, 3), dtype=f64, device=device) if compute_stress else None,
            two_body_energy=z,
            three_body_energy=z,
            charges=empty,
        )
    data = d4_reference_data(device, dtype)
    if int(numbers.min()) < 1 or int(numbers.max()) > data.max_number:
        raise ValueError(f"atomic numbers must be in 1..{data.max_number}")
    bohr, hartree = settings.bohr_in_angstrom, settings.hartree_in_ev
    pos = torch.as_tensor(positions).detach().to(device=device, dtype=dtype) / bohr
    cell_b = torch.as_tensor(cell).detach().to(device=device, dtype=dtype) / bohr if pbc else None

    # neighbour lists (no autograd): one build at the largest cutoff, the others by masking
    c2, c3 = settings.two_body_cutoff_bohr, settings.three_body_cutoff_bohr
    ccn = settings.coordination_cutoff_bohr
    do_two = abs(settings.s6) >= torch.finfo(f64).eps or abs(settings.s8) >= torch.finfo(f64).eps
    do_three = abs(settings.s9) >= torch.finfo(f64).eps
    cutoffs = [ccn, eeq.CN_CUTOFF] + [c2] * do_two + [c3] * do_three
    r_max = max(cutoffs)
    nl_all = make_builder(settings.neighbor_backend)(pos, cell_b, pbc, r_max)
    _check_coincident(nl_all, pos, cell_b)

    def restricted(r: float) -> NeighborMatrix:
        return nl_all if r >= r_max else nl_all.restrict(pos, cell_b, r)

    block, tile, pair_tile = (
        settings.centre_block_size,
        settings.neighbor_tile_size,
        settings.pair_tile_size,
    )
    n_padded = round_up(n + 1, block)
    nl_cn = restricted(ccn).pad(n_padded, CN_WIDTH_MULTIPLE)
    nl_eeq = restricted(eeq.CN_CUTOFF).pad(n_padded, CN_WIDTH_MULTIPLE)
    system = _pad_system(numbers, n_padded, data)

    # stage 1: CN, charges and C6 with autograd
    need_grad = compute_forces or compute_stress
    pos_leaf = pos.clone().requires_grad_(compute_forces)
    strain = torch.zeros((3, 3), dtype=dtype, device=device, requires_grad=compute_stress)
    deform = torch.eye(3, dtype=dtype, device=device) + strain
    pos_s = pos_leaf @ deform if pbc else pos_leaf
    cell_s = cell_b @ deform if cell_b is not None else None
    with torch.set_grad_enabled(need_grad):
        pos_p = pad_positions(pos_s, n_padded)
        cn, cn_eeq = _maybe_compiled(_coordination_numbers, settings.compile)(
            nl_cn.vectors(pos_p, cell_s),
            nl_cn.mask,
            nl_cn.indices,
            nl_eeq.vectors(pos_p, cell_s),
            nl_eeq.mask,
            nl_eeq.indices,
            system.rcov,
            system.en,
            ccn,
        )
        q = _charges(numbers, pos_s, cell_s, cell_b, cn_eeq[:n], data, charge)
        q_p = torch.nn.functional.pad(q.to(dtype), (0, n_padded - n))
        c6_q, c6_0 = _maybe_compiled(_c6_pair, settings.compile)(
            cn,
            q_p,
            system.reference_cn,
            system.reference_q,
            system.n_gaussians,
            system.zeff,
            system.hardness,
            system.species,
            system.c6_species,
        )

    # stage 2: blocked two-body pairs and ATM triples on detached C6
    pos_d = pad_positions(pos_s.detach(), n_padded)
    cell_d = None if cell_s is None else cell_s.detach()
    r4r2 = data.r4r2
    rr = 3.0 * r4r2[:, None] * r4r2[None, :]
    r0 = settings.a1 * torch.sqrt(rr) + settings.a2

    def zero():
        if need_grad:
            return zero_accumulators(n_padded, device)
        return torch.zeros((), dtype=f64, device=device)

    acc2 = acc3 = None
    if do_two:
        nl2 = restricted(c2).pad(n_padded, pair_tile)
        step2 = get_pair_step_fn(need_grad, settings.compile)
        scal2 = pair_scalars(settings.s6, settings.s8, c2, dtype, device)
        acc2 = run_pair_tiles(
            step2,
            zero(),
            c6_q.detach(),
            nl2,
            nl2.vectors(pos_d, cell_d),
            system.numbers,
            n,
            block,
            pair_tile,
            rr,
            r0,
            scal2,
        )
    if do_three:
        nl3 = restricted(c3).pad(n_padded, tile)
        # the kernel's exponent is (alpha + 2) / 3; D4 uses alp / 3
        scal3 = KernelParams(s9=settings.s9, alpha=settings.alp - 2.0, cutoff=c3, width=0.0)
        step3 = get_step_fn(need_grad, settings.compile, settings.kernel_gradient)
        acc3 = run_tiles(
            step3,
            zero(),
            (c6_0.detach(),),
            nl3,
            nl3.vectors(pos_d, cell_d),
            system.numbers,
            n,
            block,
            tile,
            r0,
            scal3.scalars(dtype, device),
        )

    def energy_of(acc) -> Tensor:
        if acc is None:
            return torch.zeros((), dtype=f64, device=device)
        return acc.energy if isinstance(acc, Accumulators) else acc

    e2, e3 = energy_of(acc2), energy_of(acc3)
    forces = stress = virial_out = None
    if need_grad:
        # stage 3: close the chain through CN, charges -> C6
        accs = [(c6, a) for c6, a in ((c6_q, acc2), (c6_0, acc3)) if a is not None]
        leaves = [pos_leaf] if compute_forces else []
        leaves += [strain] if compute_stress else []
        explicit_pos = sum(
            (a.grad_pos for _, a in accs), torch.zeros((n_padded, 3), dtype=f64, device=device)
        )
        explicit_virial = sum(
            (a.virial for _, a in accs), torch.zeros((3, 3), dtype=f64, device=device)
        )
        outputs = [c6 for c6, _ in accs if c6.requires_grad]
        grads = [a.grad_c6.to(c6.dtype) for c6, a in accs if c6.requires_grad]
        chain = [torch.zeros_like(x) for x in leaves]
        if outputs:
            chain = [
                torch.zeros_like(x) if g is None else g
                for x, g in zip(
                    leaves,
                    torch.autograd.grad(outputs, leaves, grad_outputs=grads, allow_unused=True),
                    strict=True,
                )
            ]
        if compute_forces:
            gradient = explicit_pos[:n] + chain.pop(0).to(f64)
            forces = -gradient * (hartree / bohr)
        if compute_stress:
            assert cell is not None
            virial_out = (explicit_virial + chain.pop(0).to(f64)) * hartree
            volume = torch.linalg.det(torch.as_tensor(cell).to(device=device, dtype=f64)).abs()
            stress = 0.5 * (virial_out + virial_out.T) / volume
    return D4Result(
        energy=(e2 + e3) * hartree,
        forces=forces,
        stress=stress,
        virial=virial_out,
        two_body_energy=e2 * hartree,
        three_body_energy=e3 * hartree,
        charges=q.detach().to(f64),
    )


__all__ = ["D4Result", "d4_dispersion"]
