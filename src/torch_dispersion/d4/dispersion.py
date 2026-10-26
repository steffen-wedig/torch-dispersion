"""Orchestration: ``d4_dispersion_batched(...) -> D4BatchResult`` and ``d4_dispersion(...) ->
D4Result`` (dftd4 4.3.0 ``get_dispersion``), one code path for a batch of S systems (a single
system is a batch of one).

The atoms of all systems are concatenated (system by system). Gradient strategy, as for the D3
ATM term (``core.chain``), with every stage working on the whole batch at once:

1. graph stage (autograd): positions and one strain per system -> D4 CN and EEQ CN over the flat
   neighbour list -> EEQ charges (per-system Coulomb matrices in a padded (S, n_max, n_max)
   layout, one batched bordered solve with the per-system total charges) -> charge-scaled
   reference weights -> C6(q) (two-body) and C6(q = 0) (ATM) as per-system blocks
   (S, n_max, n_max);
2. tile stage on detached C6: the two-body pair tiles run over all atoms of the batch at once
   (the kernel reads each pair's C6 from its system's block); the ATM triple tiles, when
   s9 != 0, run system by system on that system's C6(q = 0) block. Both return their energy and
   their gradients with respect to the displacement vectors and to C6;
3. one ``torch.autograd.grad`` through both C6 adds the CN and charge contributions.

Internally everything is in bohr and Hartree; the API takes and returns Å and eV. The EEQ stage
always runs in float64 (it is O(n^2) per system and its Ewald sums cancel strongly).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..atm.kernel import KernelParams, get_step_fn, triple_inputs
from ..core.batch import Batch, pad_to_systems
from ..core.chain import Strained, close_chain, forces_and_stress, graph_fn, strained
from ..core.inputs import check_coincident, check_numbers, check_pbc, to_bohr
from ..core.padding import CN_WIDTH_MULTIPLE, pad_numbers, pad_positions, padded_size
from ..core.tiles import Accumulators, TileLayout, run_tiles, zero_energy_or_accumulators
from ..neighbors import NeighborMatrix, build_flat
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
class D4BatchResult:
    energy: Tensor  # (S,) eV, two-body + three-body
    forces: Tensor | None  # (N, 3) eV/Å
    stress: Tensor | None  # (S, 3, 3) eV/Å^3, ASE convention, symmetrised; None without PBC
    virial: Tensor | None  # (S, 3, 3) eV, dE/dε; None without PBC
    two_body_energy: Tensor  # (S,) eV
    three_body_energy: Tensor  # (S,) eV
    charges: Tensor  # (N,) EEQ partial charges (e)


@dataclass
class _Tables:
    """Per-row reference tables of the flat batch padded with dummies (Z = 0, at the origin, no
    references) to ``Np`` rows, and the reference C6 of the batch's species."""

    numbers: Tensor  # (Np,)
    species: Tensor  # (Np,) index into the unique elements of the batch (0 = the dummy)
    c6_reference: Tensor  # (T, R, T, R) reference C6 of the species of the batch
    rcov: Tensor  # (Np,)
    en: Tensor  # (Np,)
    reference_cn: Tensor  # (Np, R)
    reference_q: Tensor  # (Np, R)
    n_gaussians: Tensor  # (Np, R)
    zeff: Tensor  # (Np,)
    hardness: Tensor  # (Np,)


def _tables(numbers: Tensor, n_padded: int, data: D4ReferenceData) -> _Tables:
    z, unique, species = pad_numbers(numbers, n_padded)
    alpha = data.alpha_iw[unique]
    return _Tables(
        numbers=z,
        species=species,
        c6_reference=reference_c6(alpha, alpha).permute(0, 2, 1, 3),
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
    """(D4 CN, EEQ CN) of the padded rows; fixed-shape, compiles with ``fullgraph=True``."""
    cn = d4_coordination_number(vec_cn, mask_cn, rcov, rcov[idx_cn], en, en[idx_cn], cutoff)
    return cn, eeq.eeq_coordination_number(vec_eeq, mask_eeq, rcov, rcov[idx_eeq])


def _c6_blocks(
    cn: Tensor,
    q: Tensor,
    reference_cn: Tensor,
    reference_q: Tensor,
    n_gaussians: Tensor,
    zeff: Tensor,
    hardness: Tensor,
    species: Tensor,
    c6_reference: Tensor,
    index: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-system C6 blocks (S, n_max, n_max) with the EEQ charges (two-body) and with q = 0
    (ATM). C6_ij = sum_ab w_ia w_jb C6_ref(Z_i a, Z_j b) is one matmul per system, W C6_ref W^T,
    with the weights of atom i placed in the reference slots of its species."""
    n = index.shape[0]
    n_systems, n_max = mask.shape
    n_species, n_ref = c6_reference.shape[:2]
    c6_ref = c6_reference.reshape(n_species * n_ref, n_species * n_ref)
    args = (reference_cn, reference_q, n_gaussians, zeff, hardness)
    # one-hot species slots by comparison (one_hot would specialise on the species count)
    slots = torch.arange(n_species, device=species.device)
    slot = (species[:n, None] == slots).to(cn.dtype)[:, :, None]

    def blocks(weights: Tensor) -> Tensor:
        w = (slot * weights[:n, None, :]).reshape(n, n_species * n_ref)
        w = pad_to_systems(w, index, n_systems, n_max)
        return w @ c6_ref @ w.mT

    w_q = reference_weights(cn, q, *args)
    w_0 = reference_weights(cn, torch.zeros_like(q), *args)
    return blocks(w_q), blocks(w_0)


def _charges(
    cn_eeq: Tensor,
    pos: Tensor,
    cells: Tensor | None,
    a_dir: Tensor | None,
    rad: Tensor,
    eta: Tensor,
    chi: Tensor,
    kcnchi: Tensor,
    alphas: Tensor,
    total_charge: Tensor,
    index: Tensor,
    mask: Tensor,
) -> Tensor:
    """EEQ charges (N,) of the batch, all inputs float64: per atom ``cn_eeq``, ``pos`` and the
    EEQ tables; per system ``cells``, the direct Coulomb part ``a_dir`` (PBC), the Ewald
    splitting ``alphas`` and ``total_charge``; ``index``/``mask`` give the padded layout."""
    n_systems, n_max = mask.shape

    def pad(x: Tensor, fill: float = 0.0) -> Tensor:
        return pad_to_systems(x, index, n_systems, n_max, fill)

    x = pad(eeq.electronegativity_vector(cn_eeq, chi, kcnchi))
    pos_p, rad_p, eta_p = pad(pos), pad(rad, 1.0), pad(eta)
    if cells is None:
        a = eeq.coulomb_matrix_molecular(pos_p, mask, rad_p, eta_p)
    else:
        assert a_dir is not None
        a = a_dir + eeq.coulomb_matrix_reciprocal(pos_p, cells, mask, rad_p, eta_p, alphas)
    return eeq.solve_charges(a, x, mask, total_charge).reshape(-1)[index]


def _graph(
    vec_cn: Tensor,
    mask_cn: Tensor,
    idx_cn: Tensor,
    vec_eeq: Tensor,
    mask_eeq: Tensor,
    idx_eeq: Tensor,
    rcov: Tensor,
    en: Tensor,
    cutoff: float,
    pos: Tensor,
    cells: Tensor | None,
    a_dir: Tensor | None,
    eeq_tables: tuple[Tensor, Tensor, Tensor, Tensor],
    alphas: Tensor,
    total_charge: Tensor,
    index: Tensor,
    mask: Tensor,
    c6_tables: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """The graph stage of the whole batch after the host-side EEQ setup, as one function
    (compiled with dynamic shapes, so its backward is one compiled graph too): CN and EEQ CN of
    the padded rows -> EEQ charges (N,) in float64 -> C6(q), C6(q = 0) blocks (S, n_max, n_max).

    ``eeq_tables`` are (rad, eta, chi, kcnchi) of the N atoms in float64, ``c6_tables`` the
    ``_c6_blocks`` tables before ``index``."""
    f64 = torch.float64
    cn, cn_eeq = _coordination_numbers(
        vec_cn, mask_cn, idx_cn, vec_eeq, mask_eeq, idx_eeq, rcov, en, cutoff
    )
    n = pos.shape[0]
    q = _charges(
        cn_eeq[:n].to(f64),
        pos.to(f64),
        None if cells is None else cells.to(f64),
        a_dir,
        *eeq_tables,
        alphas,
        total_charge,
        index,
        mask,
    )
    q_rows = torch.nn.functional.pad(q.to(cn.dtype), (0, cn.shape[0] - n))
    c6_q, c6_0 = _c6_blocks(cn, q_rows, *c6_tables, index, mask)
    return q, c6_q, c6_0


@dataclass(frozen=True)
class _Terms:
    """Which terms run, and the cutoff of the one neighbour list (bohr)."""

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
    """Cutoff (bohr) of the neighbour list: the largest of the D4 CN, EEQ CN, two-body (if s6
    or s8 != 0) and ATM (if s9 != 0) ranges."""
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
    """DFT-D4 dispersion (two-body rational damping + ATM) of one system. Positions and cell
    (rows = lattice vectors) in Å; ``charge`` is the total charge distributed by the EEQ model.

    Returns float64 tensors on ``settings.device``: energy (eV), forces (eV/Å), stress (eV/Å^3,
    ASE sign convention), virial dE/dε (eV); stress and virial are None without PBC.
    """
    numbers = torch.as_tensor(numbers)
    n = numbers.numel()
    if n == 0:
        f64, device = torch.float64, settings.torch_device
        zero = torch.zeros((), dtype=f64, device=device)
        periodic = compute_stress and check_pbc(pbc)
        return D4Result(
            energy=zero,
            forces=torch.zeros((0, 3), dtype=f64, device=device) if compute_forces else None,
            stress=torch.zeros((3, 3), dtype=f64, device=device) if periodic else None,
            virial=torch.zeros((3, 3), dtype=f64, device=device) if periodic else None,
            two_body_energy=zero,
            three_body_energy=zero,
            charges=torch.zeros((0,), dtype=f64, device=device),
        )
    cells = None if cell is None else torch.as_tensor(cell).reshape(1, 3, 3)
    r = d4_dispersion_batched(
        numbers, positions, cells, pbc, [n], settings, [charge], compute_forces, compute_stress
    )
    return D4Result(
        energy=r.energy[0],
        forces=r.forces,
        stress=None if r.stress is None else r.stress[0],
        virial=None if r.virial is None else r.virial[0],
        two_body_energy=r.two_body_energy[0],
        three_body_energy=r.three_body_energy[0],
        charges=r.charges,
    )


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
    equal ``d4_dispersion`` on that system alone, to rounding.

    Every stage holds the whole batch at once: the neighbour lists of all systems (one
    ``system_idx``-aware nvalchemiops call), the per-system EEQ and C6 blocks (S n_max^2).
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
    total = torch.zeros(n_systems, dtype=torch.float64, device=device)
    if charges is not None:
        total = torch.as_tensor(charges, dtype=torch.float64).reshape(-1).to(device)
        if total.numel() != n_systems:
            raise ValueError(f"{total.numel()} total charges for {n_systems} systems")
    data = d4_reference_data(device, settings.torch_dtype)
    check_numbers(numbers, data.max_number)
    pos = to_bohr(positions, settings)
    cells_b = cells_a = None
    if pbc and cells is not None:
        cells_b = to_bohr(cells, settings).reshape(n_systems, 3, 3)
        cells_a = torch.as_tensor(cells).reshape(n_systems, 3, 3)
    cutoff = neighbor_list_cutoff(settings)
    nl_all = build_flat(settings.neighbor_backend, pos, cells_b, pbc, cutoff, counts)
    batch = Batch.from_counts(counts, device)
    return _evaluate(
        numbers, pos, cells_b, cells_a, batch, nl_all, settings, data, total, compute_forces,
        compute_stress,
    )  # fmt: skip


def _evaluate(
    numbers: Tensor,
    pos: Tensor,
    cells_b: Tensor | None,
    cells_a: Tensor | None,
    batch: Batch,
    nl_all: NeighborMatrix,
    settings: D4Settings,
    data: D4ReferenceData,
    total_charge: Tensor,
    compute_forces: bool,
    compute_stress: bool,
) -> D4BatchResult:
    """The three stages of the module docstring on checked inputs in bohr."""
    device = settings.torch_device
    f64 = torch.float64
    compute_stress = compute_stress and cells_b is not None
    need_grad = compute_forces or compute_stress
    n = batch.n_atoms
    terms = _terms(settings)
    row_cells = None if cells_b is None else cells_b[batch.system]
    check_coincident(nl_all, pos, row_cells)

    def restricted(r: float) -> NeighborMatrix:
        return nl_all if r >= terms.list_cutoff else nl_all.restrict(pos, row_cells, r)

    n_padded = padded_size(n, settings.centre_block_size)
    tables = _tables(numbers, n_padded, data)
    pad_rows = (0, n_padded - n)
    system_rows = torch.nn.functional.pad(batch.system, pad_rows)
    local_rows = torch.nn.functional.pad(batch.local, pad_rows)

    # stage 1: CN, charges and C6 with autograd
    geometry = strained(pos, cells_b, compute_forces, compute_stress, batch.system)
    with torch.set_grad_enabled(need_grad):
        q, c6_q, c6_0 = _graph_stage(
            geometry, cells_b, batch, system_rows, tables, restricted, settings, data,
            total_charge,
        )  # fmt: skip

    # stage 2: blocked two-body pairs (whole batch) and ATM triples (per system), detached C6
    pos_d = pad_positions(geometry.positions.detach(), n_padded)
    cells_d = None if geometry.cell is None else geometry.cell.detach()
    rows_d = None if cells_d is None else cells_d[system_rows]
    zero = torch.zeros(batch.n_systems, dtype=f64, device=device)
    e2 = e3 = zero
    explicit_pos = torch.zeros((n, 3), dtype=f64, device=device)
    explicit_virial = torch.zeros((batch.n_systems, 3, 3), dtype=f64, device=device)
    outputs: list[Tensor] = []
    grad_outputs: list[Tensor] = []
    if terms.two_body:
        acc2 = _two_body(
            c6_q.detach(), pos_d, rows_d, system_rows, local_rows, batch, tables, restricted,
            settings, data, need_grad,
        )  # fmt: skip
        if isinstance(acc2, Accumulators):
            e2 = acc2.energy
            explicit_pos = explicit_pos + acc2.grad_pos[:n]
            explicit_virial = explicit_virial + acc2.virial
            outputs.append(c6_q)
            grad_outputs.append(acc2.grad_c6.view_as(c6_q))
        else:
            e2 = acc2
    if terms.three_body:
        nl3 = restricted(settings.three_body_cutoff_bohr)
        energies = []
        for s, (start, n_s) in enumerate(_ranges(batch.counts)):
            c6_block = c6_0[s, :n_s, :n_s]
            acc3 = _three_body(
                c6_block.detach(),
                pos_d[start : start + n_s],
                None if cells_d is None else cells_d[s],
                nl3.rows(start, n_s),
                tables.numbers[start : start + n_s],
                settings,
                data,
                need_grad,
            )
            if isinstance(acc3, Accumulators):
                energies.append(acc3.energy)
                explicit_pos[start : start + n_s] += acc3.grad_pos[:n_s]
                explicit_virial[s] += acc3.virial
                outputs.append(c6_block)
                grad_outputs.append(acc3.grad_c6[:n_s, :n_s])
            else:
                energies.append(acc3)
        e3 = torch.stack(energies)

    hartree = settings.hartree_in_ev
    forces = stress = virial = None
    if need_grad:
        # stage 3: close the chain through CN, charges -> C6
        grads = close_chain(geometry, outputs, grad_outputs, explicit_pos, explicit_virial, n)
        forces, stress, virial = forces_and_stress(*grads, cells_a, settings)
    return D4BatchResult(
        energy=(e2 + e3) * hartree,
        forces=forces,
        stress=stress,
        virial=virial,
        two_body_energy=e2 * hartree,
        three_body_energy=e3 * hartree,
        charges=q.detach().to(f64),
    )


def _ranges(counts: Sequence[int]) -> list[tuple[int, int]]:
    """(first atom, atom count) of each system."""
    starts = [sum(counts[:s]) for s in range(len(counts))]
    return list(zip(starts, counts, strict=True))


def _graph_stage(
    geometry: Strained,
    cells_b: Tensor | None,
    batch: Batch,
    system_rows: Tensor,
    tables: _Tables,
    restricted,
    settings: D4Settings,
    data: D4ReferenceData,
    total_charge: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Stage 1 on the (strained) positions: EEQ charges (N,), C6(q) and C6(q = 0) blocks."""
    n_padded = tables.numbers.shape[0]
    pos_s, cells_s = geometry.positions, geometry.cell
    ccn = settings.coordination_cutoff_bohr
    nl_cn = restricted(ccn).pad(n_padded, CN_WIDTH_MULTIPLE)
    nl_eeq = restricted(eeq.CN_CUTOFF).pad(n_padded, CN_WIDTH_MULTIPLE)
    pos_p = pad_positions(pos_s, n_padded)
    rows = None if cells_s is None else cells_s[system_rows]
    f64 = torch.float64
    z = tables.numbers[: batch.n_atoms]
    rad = data.eeq_rad.to(f64)[z]
    eeq_tables = (rad, data.eeq_eta.to(f64)[z], data.eeq_chi.to(f64)[z], data.eeq_kcnchi.to(f64)[z])
    # host-side EEQ setup (depends on the unstrained cells only): splittings and image pairs,
    # then the direct Coulomb sums, which have their own analytic backward
    alphas = torch.zeros(batch.n_systems, dtype=f64, device=pos_s.device)
    a_dir = None
    if cells_s is not None:
        assert cells_b is not None
        alphas = eeq.ewald_alphas(cells_b)
        pairs = eeq.wigner_seitz_images(pos_s.detach(), cells_b.to(f64), batch)
        pos64, cells64 = pos_s.to(f64), cells_s.to(f64)
        a_dir = eeq.direct_coulomb(pos64, cells64, rad, alphas, pairs, batch, settings.compile)
    c6_tables = (
        tables.reference_cn,
        tables.reference_q,
        tables.n_gaussians,
        tables.zeff,
        tables.hardness,
        tables.species,
        tables.c6_reference,
    )
    return graph_fn(_graph, settings.compile)(
        nl_cn.vectors(pos_p, rows),
        nl_cn.mask,
        nl_cn.indices,
        nl_eeq.vectors(pos_p, rows),
        nl_eeq.mask,
        nl_eeq.indices,
        tables.rcov,
        tables.en,
        ccn,
        pos_s,
        cells_s,
        a_dir,
        eeq_tables,
        alphas,
        total_charge,
        batch.index,
        batch.mask,
        c6_tables,
    )


def _two_body(
    c6: Tensor,
    pos_d: Tensor,
    rows_d: Tensor | None,
    system_rows: Tensor,
    local_rows: Tensor,
    batch: Batch,
    tables: _Tables,
    restricted,
    settings: D4Settings,
    data: D4ReferenceData,
    need_grad: bool,
):
    """Two-body tiles over all atoms of the batch on the detached C6(q) blocks (S, n_max,
    n_max): energy (S,) or ``Accumulators`` (per-system energy and virial, dE/dC6 flat)."""
    device, dtype = settings.torch_device, settings.torch_dtype
    n_padded = tables.numbers.shape[0]
    block, tile = settings.centre_block_size, settings.pair_tile_size
    cutoff = settings.two_body_cutoff_bohr
    nl = restricted(cutoff).pad(n_padded, tile)
    layout = TileLayout.build(nl, nl.vectors(pos_d, rows_d), tables.numbers, block, tile)
    n_max = batch.n_max
    c6_index = (system_rows * n_max + local_rows)[:, None] * n_max + local_rows[nl.indices]
    rr, r0 = rational_tables(data.r4r2, settings.a1, settings.a2)
    return run_tiles(
        get_pair_step_fn(need_grad, settings.compile),
        zero_energy_or_accumulators(need_grad, n_padded, device, batch.n_systems, c6.numel()),
        (c6.reshape(-1),),
        layout,
        batch.n_atoms,
        1,
        pair_inputs(rr, r0, layout.tiled(c6_index), layout.rows(system_rows)),
        pair_scalars(settings.s6, settings.s8, cutoff, dtype, device),
    )


def _three_body(
    c6: Tensor,
    pos: Tensor,
    cell: Tensor | None,
    nl: NeighborMatrix,
    numbers: Tensor,
    settings: D4Settings,
    data: D4ReferenceData,
    need_grad: bool,
):
    """ATM tiles of one system on its detached C6(q = 0) block (n, n): energy () or
    ``Accumulators`` of the system padded to the centre block size."""
    device, dtype = settings.torch_device, settings.torch_dtype
    n = numbers.shape[0]
    block, tile = settings.centre_block_size, settings.neighbor_tile_size
    n_padded = padded_size(n, block)
    nl = nl.pad(n_padded, tile)
    z = torch.nn.functional.pad(numbers, (0, n_padded - n))
    layout = TileLayout.build(nl, nl.vectors(pad_positions(pos, n_padded), cell), z, block, tile)
    c6_p = torch.nn.functional.pad(c6, (0, n_padded - n, 0, n_padded - n))
    _, r0 = rational_tables(data.r4r2, settings.a1, settings.a2)
    # the kernel's exponent is (alpha + 2) / 3; D4 uses alp / 3
    cutoff = settings.three_body_cutoff_bohr
    scalars = KernelParams(s9=settings.s9, alpha=settings.alp - 2.0, cutoff=cutoff, width=0.0)
    return run_tiles(
        get_step_fn(need_grad, settings.compile, settings.kernel_gradient),
        zero_energy_or_accumulators(need_grad, n_padded, device),
        (c6_p,),
        layout,
        n,
        2,
        triple_inputs(r0),
        scalars.scalars(dtype, device),
    )


__all__ = [
    "D4BatchResult",
    "D4Result",
    "d4_dispersion",
    "d4_dispersion_batched",
    "neighbor_list_cutoff",
]
