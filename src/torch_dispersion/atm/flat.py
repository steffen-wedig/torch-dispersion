"""ATM triple kernel over a flat batch of systems, with per-pair gradients.

The per-system kernel (``atm.kernel``) reads C6 from one system's (Np, Np) matrix and scatters
the far side's dE/dC6 with one-hot (B, T, Np) matmuls, so a batch had to be visited system by
system and each system padded to whole tiles. This kernel runs one tile loop over every atom of
the batch, as the two-body kernel (``pairwise.rational``) does:

- tiles are cut from the flat ATM neighbour list when they are used (``core.tiles.SliceLayout``
  with per-block widths, so a block of small molecules gets a narrow tile); the kernel builds
  the centre legs from positions, neighbour indices, lattice shifts and the centre's cell;
- C6 comes from the flattened per-system blocks (S * n_max^2) at (system, local i, local j);
  the far side (n, m) lies in the centre's system, so C6_nm is a read from the same block;
- gradients use the pair identity of ``docs/atm_pair_gradients.md``. The triangle energy is
  symmetric in its corners, and each side is a centre leg in two of the three visits of its
  triangle and the far side in one. So 3/2 times the centre-leg derivatives is exact, and the
  far side needs no derivative. Every derivative then belongs to an entry (i, j, shift) of row
  i of the full list, and like the two-body kernel:
  - the force on a centre is -2 x the sum of dE/dv over its own row (the partner entry
    (j, i, -shift) carries the same derivative with the opposite vector);
  - the virial sums v (x) dE/dv over all entries as they are;
  - dE/dC6 is an ``index_add`` of B T entries per tile into the flat C6 blocks.

Energy, virial and dE/dC6 accumulate per system. The per-atom tables and the accumulators are
dynamic dimensions, so the compiled graphs serve every system. The block size B is fixed;
full-width tiles have a static width, and narrower tiles (which vary by block) share one graph
with a dynamic width (see ``get_flat_step_fn``).
"""

from functools import cache
from typing import NamedTuple

import torch
from torch import Tensor

from ..core.tiles import Accumulators, SliceLayout
from .kernel import damping_radii, leg_derivatives, triangles


class FlatScalars(NamedTuple):
    s9: Tensor
    exponent: Tensor  # (alpha + 2) / 3
    cutoff: Tensor
    width: Tensor  # 0 = hard cutoff
    n_max: Tensor  # () int64, the side of the per-system C6 blocks


class FlatTileInputs(NamedTuple):
    centre: Tensor  # (B,) atom index of the centres
    idx_n: Tensor  # (B, T) atom index of the n sites (tile p)
    idx_m: Tensor  # (B, T) of the m sites (tile q)
    shift_n: Tensor  # (B, T, 3) integer lattice shifts
    shift_m: Tensor
    mask_n: Tensor  # (B, T) bool
    mask_m: Tensor
    system: Tensor  # (B,) system of each centre
    r0: Tensor  # (Z, Z) damping radius table, bohr
    diagonal: Tensor  # () bool: p == q, keep only n < m


def flat_scalars(
    s9: float, alpha: float, cutoff: float, width: float, n_max: int, dtype, device
) -> FlatScalars:
    """``alpha`` as in ``atm.kernel.KernelParams`` (the exponent is (alpha + 2) / 3)."""

    def t(x: float) -> Tensor:
        return torch.tensor(x, dtype=dtype, device=device)

    n_max_t = torch.tensor(n_max, dtype=torch.int64, device=device)
    return FlatScalars(
        s9=t(s9), exponent=t((alpha + 2.0) / 3.0), cutoff=t(cutoff), width=t(width), n_max=n_max_t
    )


def flat_inputs(r0: Tensor, system: Tensor):
    """``make_inputs`` of ``core.tiles.run_tiles`` (order 2) for the tile pair (p, q) of centre
    block kb of a ``SliceLayout``; ``system`` (Np,) is the system of each row."""
    diagonal = {flag: torch.tensor(flag, device=r0.device) for flag in (True, False)}

    def make(layout: SliceLayout, kb: int, tiles: tuple[int, ...]) -> FlatTileInputs:
        p, q = tiles
        nl = layout.nl
        centre = layout.centres[kb]
        return FlatTileInputs(
            centre=centre,
            idx_n=layout.entries(nl.indices, kb, p, nl.n_atoms),
            idx_m=layout.entries(nl.indices, kb, q, nl.n_atoms),
            shift_n=layout.entries(nl.shifts, kb, p, 0),
            shift_m=layout.entries(nl.shifts, kb, q, 0),
            mask_n=layout.entries(nl.mask, kb, p, False),
            mask_m=layout.entries(nl.mask, kb, q, False),
            system=system[centre],
            r0=r0,
            diagonal=diagonal[p == q],
        )

    return make


def _legs(pos: Tensor, cells: Tensor, idx: Tensor, shift: Tensor, mask: Tensor, centre: Tensor):
    """Centre legs x_j - x_i + shift @ cell_i (B, T, 3), zero in padding. The 3-term product is
    a broadcast sum so that it fuses into the pointwise kernel."""
    lattice = (shift.to(pos.dtype)[..., :, None] * cells[centre][:, None, :, :]).sum(-2)
    v = pos[idx] - pos[centre][:, None, :] + lattice
    return torch.where(mask[..., None], v, 0.0)


class _Tile(NamedTuple):
    v_n: Tensor  # (B, T, 3)
    v_m: Tensor
    c6_n: Tensor  # (B, T) index of C6_in in the flat blocks
    c6_m: Tensor


def _tile(pos, cells, local, t: FlatTileInputs, s: FlatScalars) -> _Tile:
    v_n = _legs(pos, cells, t.idx_n, t.shift_n, t.mask_n, t.centre)
    v_m = _legs(pos, cells, t.idx_m, t.shift_m, t.mask_m, t.centre)
    row = (t.system * s.n_max + local[t.centre])[:, None] * s.n_max
    return _Tile(v_n, v_m, row + local[t.idx_n], row + local[t.idx_m])


def _forward(
    sqrt_c6: Tensor,
    pos: Tensor,
    cells: Tensor,
    numbers: Tensor,
    local: Tensor,
    t: FlatTileInputs,
    s: FlatScalars,
):
    x = _tile(pos, cells, local, t, s)
    # C6_nm: rows of n's system block (the centre's), columns of m
    jk = (t.system[:, None, None] * s.n_max + local[t.idx_n][:, :, None]) * s.n_max
    jk = jk + local[t.idx_m][:, None, :]
    k = (s.s9 / 3.0) * (sqrt_c6[x.c6_n][:, :, None] * sqrt_c6[x.c6_m][:, None, :] * sqrt_c6[jk])
    r0 = damping_radii(t.r0, numbers[t.centre], numbers[t.idx_n], numbers[t.idx_m])
    terms = triangles(x.v_n, x.v_m, t.mask_n, t.mask_m, t.diagonal, r0, k, s)
    return x, terms


def energy_step(
    acc: Tensor,
    sqrt_c6: Tensor,
    inv_c6: Tensor,
    pos: Tensor,
    cells: Tensor,
    numbers: Tensor,
    local: Tensor,
    t: FlatTileInputs,
    s: FlatScalars,
) -> Tensor:
    _, terms = _forward(sqrt_c6, pos, cells, numbers, local, t, s)
    return acc.index_add(0, t.system, terms.e.to(torch.float64).sum((1, 2)))


def _pointwise(
    sqrt_c6: Tensor,
    pos: Tensor,
    cells: Tensor,
    numbers: Tensor,
    local: Tensor,
    t: FlatTileInputs,
    s: FlatScalars,
) -> tuple[Tensor, Tensor, Tensor]:
    """(e, de/du_ij, de/du_ik), all (B, T, T)."""
    _, terms = _forward(sqrt_c6, pos, cells, numbers, local, t, s)
    g_ij, g_ik = leg_derivatives(terms, s, (0, 1))
    return terms.e, g_ij, g_ik


def _reduce_step(
    acc: Accumulators,
    inv_c6: Tensor,
    pos: Tensor,
    cells: Tensor,
    local: Tensor,
    t: FlatTileInputs,
    s: FlatScalars,
    e: Tensor,
    g_ij: Tensor,
    g_ik: Tensor,
) -> Accumulators:
    """Row and column sums of the triangle derivatives into per-entry derivatives (x 3/2, the
    far side dropped), then forces from row sums, virial and dE/dC6 per entry."""
    f64 = torch.float64
    x = _tile(pos, cells, local, t, s)
    # dE/dv per entry: u = |v|^2, so dE/dv = 2 (dE/du) v
    dv_n = (3.0 * g_ij.sum(2))[..., None] * x.v_n
    dv_m = (3.0 * g_ik.sum(1))[..., None] * x.v_m
    dv_n64, dv_m64 = dv_n.to(f64), dv_m.to(f64)
    grad_pos = acc.grad_pos.index_add(0, t.centre, -2.0 * (dv_n64.sum(1) + dv_m64.sum(1)))
    virial = (x.v_n.to(f64)[..., :, None] * dv_n64[..., None, :]).sum(1)
    virial = virial + (x.v_m.to(f64)[..., :, None] * dv_m64[..., None, :]).sum(1)  # (B, 3, 3)
    # dE/dC6_x = e / (2 C6_x), x 3/2
    h_n = (0.75 * e.sum(2)) * inv_c6[x.c6_n]
    h_m = (0.75 * e.sum(1)) * inv_c6[x.c6_m]
    index = torch.cat([x.c6_n.flatten(), x.c6_m.flatten()])
    grad_c6 = acc.grad_c6.index_add(0, index, torch.cat([h_n.flatten(), h_m.flatten()]).to(f64))
    return Accumulators(
        energy=acc.energy.index_add(0, t.system, e.to(f64).sum((1, 2))),
        grad_pos=grad_pos,
        virial=acc.virial.index_add(0, t.system, virial),
        grad_c6=grad_c6,
    )


def gradient_step(
    acc: Accumulators,
    sqrt_c6: Tensor,
    inv_c6: Tensor,
    pos: Tensor,
    cells: Tensor,
    numbers: Tensor,
    local: Tensor,
    t: FlatTileInputs,
    s: FlatScalars,
) -> Accumulators:
    e, g_ij, g_ik = _pointwise(sqrt_c6, pos, cells, numbers, local, t, s)
    return _reduce_step(acc, inv_c6, pos, cells, local, t, s, e, g_ij, g_ik)


def _compile(fn):
    return torch.compile(fn, fullgraph=True, dynamic=False)


# Full-width tiles (the hot path of dense systems) run graphs with a static tile width, which
# are ~1.7x faster on a B200 than graphs with a dynamic width (12 molecular crystals with ATM:
# 4.9 s against 8.4 s). Narrower tiles (blocks of small molecules) share one graph with a
# dynamic width. Dynamo caches and counts recompilations per code object, so each variant is
# its own function; otherwise every width would be another entry towards recompile_limit (8),
# a hard error under fullgraph.
def _energy_full(*args):
    return energy_step(*args)


def _energy_narrow(*args):
    return energy_step(*args)


def _pointwise_full(*args):
    return _pointwise(*args)


def _pointwise_narrow(*args):
    return _pointwise(*args)


def _reduce_full(*args):
    return _reduce_step(*args)


def _reduce_narrow(*args):
    return _reduce_step(*args)


@cache
def _compiled(with_grad: bool, narrow: bool):
    if not with_grad:
        return _compile(_energy_narrow if narrow else _energy_full)
    # two graphs, as in atm.kernel: the per-triangle math as one pointwise kernel, then the
    # row/column reductions and scatters
    pointwise = _compile(_pointwise_narrow if narrow else _pointwise_full)
    reduce = _compile(_reduce_narrow if narrow else _reduce_full)

    def step(acc, sqrt_c6, inv_c6, pos, cells, numbers, local, t, s):
        e, g_ij, g_ik = pointwise(sqrt_c6, pos, cells, numbers, local, t, s)
        return reduce(acc, inv_c6, pos, cells, local, t, s, e, g_ij, g_ik)

    return step


def _mark_width_dynamic(t: FlatTileInputs) -> None:
    for x in (t.idx_n, t.idx_m, t.shift_n, t.shift_m, t.mask_n, t.mask_m):
        torch._dynamo.mark_dynamic(x, 1)


def get_flat_step_fn(with_grad: bool, compile: bool, full_width: int):
    """``fn(acc, sqrt_c6, inv_c6, pos, cells, numbers, local, tile_inputs, scalars) -> acc``
    for a flat batch of S systems: ``sqrt_c6``, ``inv_c6`` the flattened per-system blocks of
    sqrt|C6| and 1/C6 (0 where |C6| < eps), then the per-atom tables of the padded rows (Np):
    positions (Np, 3), the cell of each row (Np, 3, 3; zeros without PBC), atomic numbers and
    the index of each atom within its system. Energy only (``acc`` (S,)) or with gradients
    (``acc`` an ``Accumulators`` with energy (S,), virial (S, 3, 3) and dE/dC6 over the flat
    blocks). ``full_width`` is the layout's maximum tile width: tiles of that width run a
    static-shape graph, narrower ones a graph with a dynamic width."""
    if not compile:
        return gradient_step if with_grad else energy_step
    full, narrow = _compiled(with_grad, False), _compiled(with_grad, True)

    def step(acc, sqrt_c6, inv_c6, pos, cells, numbers, local, t: FlatTileInputs, s):
        if t.idx_n.shape[1] == full_width:
            return full(acc, sqrt_c6, inv_c6, pos, cells, numbers, local, t, s)
        _mark_width_dynamic(t)
        return narrow(acc, sqrt_c6, inv_c6, pos, cells, numbers, local, t, s)

    return step


def c6_tables(c6: Tensor) -> tuple[Tensor, Tensor]:
    """(sqrt|C6|, 1/C6) of flattened C6 blocks, 0 where |C6| < eps (simple-dftd3 skips those
    triangles)."""
    eps = torch.finfo(c6.dtype).eps
    ok = c6.abs() >= eps
    safe = torch.where(ok, c6, 1.0)
    return torch.where(ok, torch.sqrt(safe.abs()), 0.0), torch.where(ok, 1.0 / safe, 0.0)
