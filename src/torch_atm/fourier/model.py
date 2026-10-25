"""Orchestration: ``fourier_atm(...) -> FourierATMResult``.

    E = s9 * E_trace(A(q), a, mu)                         all triangles, undamped, separable C9
      + sum_{legs <= r_s} [k_D3 fdmp - k_CP] ang s s s   real-space correction to exact D3

1. graph stage: positions and strain -> CN -> reference weights -> C6 (Np, Np) and the CP
   factors a (Np, R), with autograd;
2. short-range triple stage on detached C6 and a (``torch_atm.atm.run_tiles``), accumulating
   dE/dR, the virial, dE/dC6 and dE/da;
3. the dense trace with autograd through a, the positions and the cell, and one
   ``torch.autograd.grad`` that closes both chains.

Internally bohr and Hartree; the API takes and returns Å and eV.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import torch
from torch import Tensor

from ..atm import (
    CN_WIDTH_MULTIPLE,
    _check_coincident,
    check_pbc,
    pad_positions,
    pad_system,
    round_up,
    run_tiles,
)
from ..c6 import pairwise_c6, reference_weights
from ..coordination import coordination_number, smooth_coordination_number
from ..data import check_numbers, reference_data
from ..kernel import KernelParams
from ..neighbors import make_builder
from ..settings import FourierATMSettings
from .cp import CPFit, fit_c9
from .dipole import (
    EwaldSetup,
    grid_for_cell,
    molecular_dipole_matrix,
    q_grid,
    singular_correction,
    trace_energy,
)
from .kernel import SRAccumulators, SRScalars, get_sr_step_fn, zero_sr_accumulators
from .trace_grad import TraceGradients, trace_energy_and_grad


@dataclass
class FourierATMResult:
    """Energy (eV), forces (eV/Å), stress (eV/Å^3, ASE sign), virial dE/dε (eV) as in
    ``ATMResult``, plus the two parts of the energy (eV) and the q-grid used."""

    energy: Tensor
    forces: Tensor | None
    stress: Tensor | None
    virial: Tensor | None
    trace_energy: Tensor
    short_range_energy: Tensor
    q_grid: tuple[int, int, int] | None
    cp_max_rel_error: float


def _stage1(
    positions: Tensor,
    cell: Tensor | None,
    nl,
    rcov: Tensor,
    ref_cn: Tensor,
    ref_mask: Tensor,
    species: Tensor,
    c6_species: Tensor,
    numbers: Tensor,
    factors: Tensor,
    cutoff: float,
    smooth: bool,
) -> tuple[Tensor, Tensor]:
    """(C6 (Np, Np), a (Np, R)) of the padded system."""
    vec = nl.vectors(positions, cell)
    cn_fn = smooth_coordination_number if smooth else coordination_number
    cn = cn_fn(vec, nl.mask, rcov, rcov[nl.indices], cutoff)
    weights = reference_weights(cn, ref_cn, ref_mask)
    a = torch.einsum("ip,ipr->ir", weights, factors[numbers])
    return pairwise_c6(weights, species, c6_species), a


@cache
def _compiled_stage1():
    return torch.compile(_stage1, fullgraph=True, dynamic=True)


def ewald_parameters(real_cutoff: float, accuracy: float) -> tuple[float, float]:
    """(g, k_cutoff) for a real-space cutoff and a target accuracy."""
    eta = math.sqrt(-math.log(accuracy))
    g = eta / real_cutoff
    return g, 2.0 * g * eta


def long_range_energy(
    positions: Tensor,
    cell: Tensor | None,
    a: Tensor,
    mu: Tensor,
    nl=None,
    g: float = 0.0,
    k_cutoff: float = 0.0,
    grid: tuple[int, int, int] | None = None,
    correct_singularity: bool = True,
    memory_budget: float = 2e9,
) -> Tensor:
    """Undamped ATM energy of all triangles (Hartree, without s9) for C9 = sum mu a a a.

    Molecules (``cell`` None): one real dipole matrix. Periodic: Bloch-Ewald matrices on the
    shifted ``grid``, real-space pairs from ``nl``. Positions and cell in bohr; float64.
    """
    f64 = torch.float64
    positions = positions.to(f64)
    if cell is None:
        mat = molecular_dipole_matrix(positions)[None]
        return trace_energy(mat, torch.ones(1, dtype=f64, device=a.device), a, mu)
    assert nl is not None and grid is not None
    fractions, weights = q_grid(grid, positions.device)
    setup = EwaldSetup(positions, cell.to(f64), nl, g, k_cutoff)
    chunk = _chunk_size(setup, fractions.shape[0], mu.shape[0], memory_budget)
    energy = torch.zeros((), dtype=f64, device=positions.device)
    for start in range(0, fractions.shape[0], chunk):
        f, w = fractions[start : start + chunk], weights[start : start + chunk]
        energy = energy + trace_energy(setup.matrices(f), w, a, mu)
    if correct_singularity:
        energy = energy - singular_correction(setup, fractions, weights, a, mu) / 18.0
    return energy


def _chunk_size(setup: EwaldSetup, n_q: int, rank: int, memory_budget: float) -> int:
    """q-points per batch that bound the memory of the batched matrices and traces."""
    per_q = setup.bytes_per_q() + 16 * 3 * rank * (3 * setup.n) ** 2
    return max(1, min(n_q, int(memory_budget // per_q)))


def long_range_energy_analytic(
    positions: Tensor,
    cell: Tensor,
    a: Tensor,
    mu: Tensor,
    nl,
    g: float,
    k_cutoff: float,
    grid: tuple[int, int, int],
    memory_budget: float = 2e9,
) -> tuple[Tensor, TraceGradients]:
    """Periodic undamped energy (without s9) split for the analytic backend: the grid trace
    with explicit gradients (no autograd), and the q -> 0 correction as a differentiable
    tensor (autograd through ``a``, positions and cell). Energy = correction + grads.energy."""
    f64 = torch.float64
    positions, cell = positions.to(f64), cell.to(f64)
    fractions, weights = q_grid(grid, positions.device)
    with torch.no_grad():
        setup = EwaldSetup(positions.detach(), cell.detach(), nl, g, k_cutoff)
        chunk = _chunk_size(setup, fractions.shape[0], mu.shape[0], memory_budget)
        grads = trace_energy_and_grad(setup, fractions, weights, a.detach(), mu, chunk)
    graph_setup = EwaldSetup(positions, cell, nl, g, k_cutoff)
    correction = -singular_correction(graph_setup, fractions, weights, a, mu) / 18.0
    return correction, grads


def separable_c9(
    numbers: Tensor,
    positions: Tensor,
    cell: Tensor | None,
    pbc: bool,
    settings: FourierATMSettings,
) -> tuple[Tensor, Tensor]:
    """(a (N, R), mu (R,)) of the separable C9 for a structure (Å), float64, no autograd."""
    device, dtype = settings.torch_device, settings.torch_dtype
    numbers = torch.as_tensor(numbers).to(device=device, dtype=torch.int64)
    n = numbers.shape[0]
    data = reference_data(device, dtype)
    fit = fit_c9(tuple(sorted(set(numbers.tolist()))), settings.cp_rank)
    bohr = settings.bohr_in_angstrom
    pos = torch.as_tensor(positions).to(device=device, dtype=dtype) / bohr
    cell_b = torch.as_tensor(cell).to(device=device, dtype=dtype) / bohr if pbc else None
    nl = make_builder(settings.neighbor_backend)(pos, cell_b, pbc, settings.cn_cutoff)
    system = pad_system(numbers, n + 1, data)
    _, a = _stage1(
        pad_positions(pos, n + 1),
        cell_b,
        nl.pad(n + 1, CN_WIDTH_MULTIPLE),
        system.rcov,
        system.ref_cn,
        system.ref_mask,
        system.species,
        system.c6_species,
        system.numbers,
        fit.factors.to(device=device, dtype=dtype),
        settings.cn_cutoff,
        settings.coordination == "smooth_cut",
    )
    return a[:n].to(torch.float64), fit.weights.to(device)


def fourier_atm(
    numbers: Tensor,
    positions: Tensor,
    cell: Tensor | None,
    pbc: bool | Sequence[bool] | Tensor,
    settings: FourierATMSettings,
    compute_forces: bool = True,
    compute_stress: bool = True,
) -> FourierATMResult:
    """Cutoff-free D3 ATM three-body dispersion. Positions and cell (rows) in Å."""
    pbc = check_pbc(pbc)
    device, dtype = settings.torch_device, settings.torch_dtype
    f64 = torch.float64
    numbers = torch.as_tensor(numbers).to(device=device, dtype=torch.int64)
    n = numbers.shape[0]
    if pbc and cell is None:
        raise ValueError("a cell is required with pbc")
    compute_stress = compute_stress and pbc
    need_grad = compute_forces or compute_stress
    data = reference_data(device, dtype)
    check_numbers(numbers, data)
    fit: CPFit = fit_c9(tuple(sorted(set(numbers.tolist()))), settings.cp_rank)
    zero = torch.zeros((), dtype=f64, device=device)
    if settings.s9 == 0.0 or n == 0:
        return FourierATMResult(
            energy=zero,
            forces=torch.zeros((n, 3), dtype=f64, device=device) if compute_forces else None,
            stress=torch.zeros((3, 3), dtype=f64, device=device) if compute_stress else None,
            virial=torch.zeros((3, 3), dtype=f64, device=device) if compute_stress else None,
            trace_energy=zero,
            short_range_energy=zero,
            q_grid=None,
            cp_max_rel_error=fit.max_rel_error,
        )

    bohr, hartree = settings.bohr_in_angstrom, settings.hartree_in_ev
    pos = torch.as_tensor(positions).detach().to(device=device, dtype=dtype) / bohr
    cell_b = torch.as_tensor(cell).detach().to(device=device, dtype=dtype) / bohr if pbc else None

    # neighbour lists: CN, short-range triangles and (periodic) the Ewald real-space pairs
    r_s, r_cn = settings.short_range_cutoff_bohr, settings.cn_cutoff
    r_ew = settings.ewald_real_cutoff_bohr
    r_max = max(r_s, r_cn, r_ew if pbc else 0.0)
    builder = make_builder(settings.neighbor_backend)
    nl_all = builder(pos, cell_b, pbc, r_max)
    _check_coincident(nl_all, pos, cell_b)

    def restricted(cutoff: float):
        return nl_all if cutoff >= r_max else nl_all.restrict(pos, cell_b, cutoff)

    block, tile = settings.centre_block_size, settings.neighbor_tile_size
    n_padded = round_up(n + 1, block)
    nl_cn = restricted(r_cn).pad(n_padded, CN_WIDTH_MULTIPLE)
    nl_sr = restricted(r_s).pad(n_padded, tile)
    system = pad_system(numbers, n_padded, data)
    factors = fit.factors.to(device=device, dtype=dtype)  # rows Z = 0 .. max(Z), zero for 0
    mu = fit.weights.to(device=device, dtype=dtype)

    # stage 1: CN -> weights -> C6 and a, with autograd
    pos_leaf = pos.clone().requires_grad_(compute_forces)
    strain = torch.zeros((3, 3), dtype=dtype, device=device, requires_grad=compute_stress)
    deform = torch.eye(3, dtype=dtype, device=device) + strain
    pos_s = pos_leaf @ deform if pbc else pos_leaf
    cell_s = cell_b @ deform if cell_b is not None else None
    stage1 = _compiled_stage1() if settings.compile else _stage1
    with torch.set_grad_enabled(need_grad):
        c6, a = stage1(
            pad_positions(pos_s, n_padded),
            cell_s,
            nl_cn,
            system.rcov,
            system.ref_cn,
            system.ref_mask,
            system.species,
            system.c6_species,
            system.numbers,
            factors,
            r_cn,
            settings.coordination == "smooth_cut",
        )

    # stage 2: short-range correction on detached C6 and a
    c6_leaf, a_leaf = c6.detach(), a.detach()
    vec = nl_sr.vectors(
        pad_positions(pos_s.detach(), n_padded), None if cell_s is None else cell_s.detach()
    )
    base = KernelParams(
        s9=settings.s9, alpha=settings.alpha, cutoff=r_s, width=settings.short_range_width_bohr
    ).scalars(dtype, device)
    scalars = SRScalars(base=base, mu=mu)
    step = get_sr_step_fn(need_grad, settings.compile, settings.kernel_gradient)
    acc: Tensor | SRAccumulators
    acc = (
        zero_sr_accumulators(n_padded, settings.cp_rank, device)
        if need_grad
        else torch.zeros((), dtype=f64, device=device)
    )
    r0_table = settings.rs9 * data.r0_vdw
    acc = run_tiles(
        step,
        acc,
        (c6_leaf, a_leaf),
        nl_sr,
        vec,
        system.numbers,
        system.n,
        block,
        tile,
        r0_table,
        scalars,
    )
    e_sr = acc.energy if isinstance(acc, SRAccumulators) else acc

    # stage 3: dense trace (float64); gradients by autograd through a, positions and cell, or
    # (``trace_gradient="analytic"``, periodic) explicitly, with only the q -> 0 correction on
    # autograd
    grid = None
    explicit: TraceGradients | None = None
    mu64 = fit.weights.to(device)
    with torch.set_grad_enabled(need_grad):
        if pbc:
            assert cell_s is not None and cell_b is not None
            grid = settings.q_grid or grid_for_cell(cell_b, settings.supercell_length_bohr)
            g, k_cut = ewald_parameters(r_ew, settings.ewald_accuracy)
            nl_ew = restricted(r_ew)
            if settings.trace_gradient == "analytic" and need_grad:
                e_tr, explicit = long_range_energy_analytic(
                    pos_s, cell_s, a[:n], mu64, nl_ew, g, k_cut, grid
                )
            else:
                e_tr = long_range_energy(pos_s, cell_s, a[:n], mu64, nl_ew, g, k_cut, grid)
        else:
            e_tr = long_range_energy(pos_s, None, a[:n], mu64)
        e_tr = settings.s9 * e_tr

    forces = stress = virial_out = None
    if need_grad:
        assert isinstance(acc, SRAccumulators)
        grad_pos, virial, grad_a = acc.grad_pos[:n], acc.virial, acc.grad_a
        if explicit is not None:
            s9 = settings.s9
            grad_pos = grad_pos + s9 * explicit.grad_pos
            virial = virial + s9 * explicit.virial
            grad_a = grad_a.clone()
            grad_a[:n] += s9 * explicit.grad_a
        leaves = [pos_leaf] if compute_forces else []
        leaves += [strain] if compute_stress else []
        total = e_tr + (c6 * acc.grad_c6.to(c6.dtype)).sum() + (a * grad_a.to(a.dtype)).sum()
        grads = torch.autograd.grad(total, leaves, allow_unused=True)
        grads = [
            torch.zeros_like(x) if g is None else g for x, g in zip(leaves, grads, strict=True)
        ]
        if compute_forces:
            gradient = grad_pos + grads.pop(0).to(f64)
            forces = -gradient * (hartree / bohr)
        if compute_stress:
            assert cell is not None
            virial_out = (virial + grads.pop(0).to(f64)) * hartree
            volume = torch.linalg.det(torch.as_tensor(cell).to(device=device, dtype=f64)).abs()
            stress = 0.5 * (virial_out + virial_out.T) / volume
    if explicit is not None:
        e_tr = e_tr + settings.s9 * explicit.energy
    e_tr = e_tr.detach()
    return FourierATMResult(
        energy=(e_tr + e_sr) * hartree,
        forces=forces,
        stress=stress,
        virial=virial_out,
        trace_energy=e_tr * hartree,
        short_range_energy=e_sr * hartree,
        q_grid=grid,
        cp_max_rel_error=fit.max_rel_error,
    )
