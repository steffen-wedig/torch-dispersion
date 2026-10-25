"""Orchestration: ``d4_dispersion(...) -> D4Result`` (dftd4 4.3.0 ``get_dispersion``).

Gradient strategy, as for the D3 ATM term (``core.chain``):

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

import torch
from torch import Tensor

from ..atm.kernel import KernelParams, get_step_fn, triple_inputs
from ..core.c6 import pairwise_c6
from ..core.chain import close_chain, forces_and_stress, graph_fn, strained, zero_outputs
from ..core.inputs import check_coincident, check_numbers, prepare
from ..core.padding import CN_WIDTH_MULTIPLE, pad_numbers, pad_positions, padded_size
from ..core.tiles import Accumulators, TileLayout, run_tiles, zero_energy_or_accumulators
from ..neighbors import NeighborMatrix, make_builder
from ..pairwise.rational import get_pair_step_fn, pair_inputs, pair_scalars, rational_tables
from ..settings import D4Settings
from . import eeq
from .data import D4ReferenceData, d4_reference_data, reference_c6
from .model import d4_coordination_number, reference_weights


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
    z, unique, species = pad_numbers(numbers, n_padded)
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


@dataclass(frozen=True)
class _Terms:
    """Which terms run, and the cutoff of the one neighbour list per system (bohr)."""

    two_body: bool
    three_body: bool
    list_cutoff: float


def _terms(settings: D4Settings) -> _Terms:
    eps = torch.finfo(torch.float64).eps
    two = abs(settings.s6) >= eps or abs(settings.s8) >= eps
    three = abs(settings.s9) >= eps
    cutoffs = [settings.coordination_cutoff_bohr, eeq.CN_CUTOFF]
    cutoffs += [settings.two_body_cutoff_bohr] * two + [settings.three_body_cutoff_bohr] * three
    return _Terms(two_body=two, three_body=three, list_cutoff=max(cutoffs))


def neighbor_list_cutoff(settings: D4Settings) -> float:
    """Cutoff (bohr) of the neighbour list ``evaluate_system`` takes: the largest of the D4 CN,
    EEQ CN, two-body (if s6 or s8 != 0) and ATM (if s9 != 0) ranges."""
    return _terms(settings).list_cutoff


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
    inputs = prepare(numbers, positions, cell, pbc, settings)
    device = settings.torch_device
    if inputs.numbers.shape[0] == 0:
        energy, forces, stress, virial = zero_outputs(
            0, compute_forces, compute_stress and inputs.pbc, device
        )
        return D4Result(
            energy=energy,
            forces=forces,
            stress=stress,
            virial=virial,
            two_body_energy=energy,
            three_body_energy=energy,
            charges=torch.zeros((0,), dtype=torch.float64, device=device),
        )
    check_numbers(inputs.numbers, d4_reference_data(device, settings.torch_dtype).max_number)
    # neighbour list (no autograd): one build at the largest cutoff, the others by masking
    builder = make_builder(settings.neighbor_backend)
    nl_all = builder(inputs.positions, inputs.cell, inputs.pbc, neighbor_list_cutoff(settings))
    return evaluate_system(
        inputs.numbers,
        inputs.positions,
        inputs.cell,
        cell,
        nl_all,
        settings,
        charge,
        compute_forces,
        compute_stress,
    )


def evaluate_system(
    numbers: Tensor,
    pos: Tensor,
    cell_b: Tensor | None,
    cell: Tensor | None,
    nl_all: NeighborMatrix,
    settings: D4Settings,
    charge: float,
    compute_forces: bool,
    compute_stress: bool,
) -> D4Result:
    """D4 of one system from checked inputs: int64 ``numbers`` (n >= 1) on the settings device,
    positions and cell (None without PBC) in bohr from ``core.inputs.to_bohr``, the cell in Å
    (for the volume), and a neighbour list of ``pos`` at ``neighbor_list_cutoff(settings)``.

    Runs the three stages of the module docstring. Every tensor is sized by this one system, so
    the batched TorchSim model (``torch_dispersion.d4.batched``) calls it once per system."""
    device, dtype = settings.torch_device, settings.torch_dtype
    data = d4_reference_data(device, dtype)
    compute_stress = compute_stress and cell_b is not None
    need_grad = compute_forces or compute_stress
    n = numbers.shape[0]
    terms = _terms(settings)
    check_coincident(nl_all, pos, cell_b)

    def restricted(r: float) -> NeighborMatrix:
        return nl_all if r >= terms.list_cutoff else nl_all.restrict(pos, cell_b, r)

    n_padded = padded_size(n, settings.centre_block_size)
    system = _pad_system(numbers, n_padded, data)

    # stage 1: CN, charges and C6 with autograd
    geometry = strained(pos, cell_b, compute_forces, compute_stress)
    with torch.set_grad_enabled(need_grad):
        q, c6_q, c6_0 = _graph_stage(
            system, geometry.positions, geometry.cell, cell_b, restricted, settings, data, charge
        )

    # stage 2: blocked two-body pairs and ATM triples on detached C6
    pos_d = pad_positions(geometry.positions.detach(), n_padded)
    cell_d = None if geometry.cell is None else geometry.cell.detach()
    acc2, acc3 = _tile_stage(
        system, pos_d, cell_d, c6_q.detach(), c6_0.detach(), restricted, terms, settings, data,
        need_grad,
    )  # fmt: skip

    def energy_of(acc) -> Tensor:
        if acc is None:
            return torch.zeros((), dtype=torch.float64, device=device)
        return acc.energy if isinstance(acc, Accumulators) else acc

    hartree = settings.hartree_in_ev
    e2, e3 = energy_of(acc2), energy_of(acc3)
    forces = stress = virial = None
    if need_grad:
        # stage 3: close the chain through CN, charges -> C6
        terms_run = [(c6, acc) for c6, acc in ((c6_q, acc2), (c6_0, acc3)) if acc is not None]
        f64 = torch.float64
        explicit_pos = sum(
            (acc.grad_pos for _, acc in terms_run),
            torch.zeros((n_padded, 3), dtype=f64, device=device),
        )
        explicit_virial = sum(
            (acc.virial for _, acc in terms_run), torch.zeros((3, 3), dtype=f64, device=device)
        )
        grads = close_chain(
            geometry,
            [c6 for c6, _ in terms_run],
            [acc.grad_c6 for _, acc in terms_run],
            explicit_pos,
            explicit_virial,
            n,
        )
        forces, stress, virial = forces_and_stress(*grads, cell, settings)
    return D4Result(
        energy=(e2 + e3) * hartree,
        forces=forces,
        stress=stress,
        virial=virial,
        two_body_energy=e2 * hartree,
        three_body_energy=e3 * hartree,
        charges=q.detach().to(torch.float64),
    )


def _graph_stage(
    system: _System,
    pos_s: Tensor,
    cell_s: Tensor | None,
    cell_b: Tensor | None,
    restricted,
    settings: D4Settings,
    data: D4ReferenceData,
    charge: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Stage 1 on the (strained) positions: EEQ charges (n,), C6(q) and C6(q = 0) (Np, Np)."""
    n, n_padded = system.n, system.numbers.shape[0]
    ccn = settings.coordination_cutoff_bohr
    nl_cn = restricted(ccn).pad(n_padded, CN_WIDTH_MULTIPLE)
    nl_eeq = restricted(eeq.CN_CUTOFF).pad(n_padded, CN_WIDTH_MULTIPLE)
    pos_p = pad_positions(pos_s, n_padded)
    cn, cn_eeq = graph_fn(_coordination_numbers, settings.compile)(
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
    q = _charges(system.numbers[:n], pos_s, cell_s, cell_b, cn_eeq[:n], data, charge)
    q_p = torch.nn.functional.pad(q.to(settings.torch_dtype), (0, n_padded - n))
    c6_q, c6_0 = graph_fn(_c6_pair, settings.compile)(
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
    return q, c6_q, c6_0


def _tile_stage(
    system: _System,
    pos_d: Tensor,
    cell_d: Tensor | None,
    c6_q: Tensor,
    c6_0: Tensor,
    restricted,
    terms: _Terms,
    settings: D4Settings,
    data: D4ReferenceData,
    need_grad: bool,
):
    """Stage 2 on detached C6: the two-body and ATM accumulators (None for a skipped term)."""
    device, dtype = settings.torch_device, settings.torch_dtype
    n, n_padded = system.n, system.numbers.shape[0]
    block = settings.centre_block_size
    rr, r0 = rational_tables(data.r4r2, settings.a1, settings.a2)

    def layout(cutoff: float, tile: int) -> TileLayout:
        nl = restricted(cutoff).pad(n_padded, tile)
        return TileLayout.build(nl, nl.vectors(pos_d, cell_d), system.numbers, block, tile)

    acc2 = acc3 = None
    if terms.two_body:
        c2, tile2 = settings.two_body_cutoff_bohr, settings.pair_tile_size
        acc2 = run_tiles(
            get_pair_step_fn(need_grad, settings.compile),
            zero_energy_or_accumulators(need_grad, n_padded, device),
            (c6_q,),
            layout(c2, tile2),
            n,
            1,
            pair_inputs(rr, r0),
            pair_scalars(settings.s6, settings.s8, c2, dtype, device),
        )
    if terms.three_body:
        c3, tile3 = settings.three_body_cutoff_bohr, settings.neighbor_tile_size
        # the kernel's exponent is (alpha + 2) / 3; D4 uses alp / 3
        scal3 = KernelParams(s9=settings.s9, alpha=settings.alp - 2.0, cutoff=c3, width=0.0)
        acc3 = run_tiles(
            get_step_fn(need_grad, settings.compile, settings.kernel_gradient),
            zero_energy_or_accumulators(need_grad, n_padded, device),
            (c6_0,),
            layout(c3, tile3),
            n,
            2,
            triple_inputs(r0),
            scal3.scalars(dtype, device),
        )
    return acc2, acc3


__all__ = ["D4Result", "d4_dispersion", "evaluate_system", "neighbor_list_cutoff"]
