"""Fixed-shape two-body kernel with rational (Becke-Johnson) damping, as in DFT-D3(BJ) and D4.

The kernel is independent of the C6 model: it takes the per-system C6 blocks and the (Z, Z)
tables of ``rational_tables``. One call evaluates B centres i against one T-wide tile of their
neighbour lists (full list, both i -> j and j -> i, so every pair carries 1/2):

    E = -1/2 sum_{i, (j, T)} C6_ij [s6 / (r^6 + R0^6) + s8 rr_ij / (r^8 + R0^8)],
    rr_ij = 3 r4r2_i r4r2_j,  R0 = a1 sqrt(rr_ij) + a2,

for eps <= r^2 <= cutoff^2 (dftd4 and simple-dftd3 ``rational.f90``).

The centres may belong to different systems of a flat batch. The tiles come from a
``core.tiles.SliceLayout``: the kernel builds each tile's displacement vectors itself from the
positions, the tile's neighbour indices and lattice shifts and the centre's cell, and reads
C6_ij from the flattened per-system blocks (S * n_max^2) at (system, local i, local j), so no
per-entry table of the whole (long, 60 bohr) list is ever materialised. Energy and virial are
accumulated per system.

Like the ATM kernel, the heavy tensors are (B, T) with B and T fixed by the settings, the
per-atom tables and the accumulators are the only system-sized inputs (marked dynamic), and
parameters are 0-d tensors. Gradients are analytic: with u = r^2, de/du = -1/2 C6 (-3 s6 u^2
t6^2 - 4 s8 rr u^3 t8^2), and dE/dC6_ij is the geometric factor itself.
"""

from functools import cache
from typing import NamedTuple

import torch
from torch import Tensor

from ..core.tiles import Accumulators, SliceLayout


class PairScalars(NamedTuple):
    s6: Tensor
    s8: Tensor
    cutoff: Tensor
    n_max: Tensor  # () int64, the side of the per-system C6 blocks


class PairTileInputs(NamedTuple):
    centre: Tensor  # (B,) atom index of the centres
    idx: Tensor  # (B, T) atom index of the neighbours
    shift: Tensor  # (B, T, 3) integer lattice shifts
    mask: Tensor  # (B, T) bool
    system: Tensor  # (B,) system of each centre
    rr: Tensor  # (Z, Z) 3 r4r2_i r4r2_j
    r0: Tensor  # (Z, Z) a1 sqrt(rr) + a2


def rational_tables(r4r2: Tensor, a1: float, a2: float) -> tuple[Tensor, Tensor]:
    """(rr, R0) (Z, Z) from the r4r2 table: rr = 3 r4r2_i r4r2_j, R0 = a1 sqrt(rr) + a2."""
    rr = 3.0 * r4r2[:, None] * r4r2[None, :]
    return rr, a1 * torch.sqrt(rr) + a2


def pair_inputs(rr: Tensor, r0: Tensor, system: Tensor):
    """``make_inputs`` of ``core.tiles.run_tiles`` (order 1) for tile p of centre block kb of a
    ``SliceLayout``; ``system`` (Np,) is the system of each row."""

    def make(layout: SliceLayout, kb: int, tiles: tuple[int, ...]) -> PairTileInputs:
        (p,) = tiles
        nl = layout.nl
        centre = layout.centres[kb]
        return PairTileInputs(
            centre=centre,
            idx=layout.entries(nl.indices, kb, p, nl.n_atoms),
            shift=layout.entries(nl.shifts, kb, p, 0),
            mask=layout.entries(nl.mask, kb, p, False),
            system=system[centre],
            rr=rr,
            r0=r0,
        )

    return make


def pair_scalars(s6: float, s8: float, cutoff: float, n_max: int, dtype, device) -> PairScalars:
    def t(x: float) -> Tensor:
        return torch.tensor(x, dtype=dtype, device=device)

    n_max_t = torch.tensor(n_max, dtype=torch.int64, device=device)
    return PairScalars(s6=t(s6), s8=t(s8), cutoff=t(cutoff), n_max=n_max_t)


def _tile(
    pos: Tensor, cells: Tensor, numbers: Tensor, local: Tensor, t: PairTileInputs, s: PairScalars
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(v (B, T, 3), z_centre (B, 1), z (B, T), C6 index (B, T)) of one tile: the displacement
    x_j - x_i + shift @ cell_i (zero in padding), the atomic numbers and the entry of C6_ij in
    the flattened per-system blocks. The 3-term products are written out as broadcast sums so
    that they fuse into the pointwise kernel (a batched matmul with an inner size of 3 would be
    a separate, slow GEMM call)."""
    shift = t.shift.to(pos.dtype)
    lattice = (shift[..., :, None] * cells[t.centre][:, None, :, :]).sum(-2)
    v = pos[t.idx] - pos[t.centre][:, None, :] + lattice
    v = torch.where(t.mask[..., None], v, 0.0)
    row = t.system * s.n_max + local[t.centre]
    c6_index = row[:, None] * s.n_max + local[t.idx]
    return v, numbers[t.centre][:, None], numbers[t.idx], c6_index


def _geometry(
    v: Tensor, z_centre: Tensor, z: Tensor, t: PairTileInputs, s: PairScalars
) -> tuple[Tensor, Tensor]:
    """Per pair (g, dg/du) with E = sum C6 g; zero outside the cutoff and in padding."""
    eps = torch.finfo(v.dtype).eps
    u = v.square().sum(-1)
    valid = t.mask & (u <= s.cutoff * s.cutoff) & (u >= eps)
    u = torch.where(valid, u, 1.0)
    rr, r0 = t.rr[z_centre, z], t.r0[z_centre, z]
    r0_2 = r0 * r0
    r0_6 = r0_2 * r0_2 * r0_2
    u2 = u * u
    t6 = 1.0 / (u2 * u + r0_6)
    t8 = 1.0 / (u2 * u2 + r0_6 * r0_2)
    g = -0.5 * (s.s6 * t6 + s.s8 * rr * t8)
    dg = 0.5 * (3.0 * s.s6 * u2 * t6 * t6 + 4.0 * s.s8 * rr * u2 * u * t8 * t8)
    return torch.where(valid, g, 0.0), torch.where(valid, dg, 0.0)


def energy_step(
    acc: Tensor,
    c6: Tensor,
    pos: Tensor,
    cells: Tensor,
    numbers: Tensor,
    local: Tensor,
    t: PairTileInputs,
    s: PairScalars,
) -> Tensor:
    v, z_centre, z, c6_index = _tile(pos, cells, numbers, local, t, s)
    g, _ = _geometry(v, z_centre, z, t, s)
    return acc.index_add(0, t.system, (c6[c6_index] * g).to(torch.float64).sum(1))


def gradient_step(
    acc: Accumulators,
    c6: Tensor,
    pos: Tensor,
    cells: Tensor,
    numbers: Tensor,
    local: Tensor,
    t: PairTileInputs,
    s: PairScalars,
) -> Accumulators:
    """Tile energy and gradients.

    The list is full and symmetric: pair (i, j, T) in row i has the partner (j, i, -T) in row j
    with the same C6, damping and switching and the opposite vector, so the partner's dE/dv is
    the negative of this one's. The force on a centre is therefore -2 x the sum of dE/dv over its
    own row, with no scatter to the neighbours; the virial sums both copies as they are."""
    f64 = torch.float64
    v, z_centre, z, c6_index = _tile(pos, cells, numbers, local, t, s)
    g, dg = _geometry(v, z_centre, z, t, s)
    c6p = c6[c6_index]
    dv = (2.0 * c6p * dg)[..., None] * v  # dE/dv, (B, T, 3)
    dv64 = dv.to(f64)
    grad_pos = acc.grad_pos.index_add(0, t.centre, -2.0 * dv64.sum(1))
    grad_c6 = acc.grad_c6.index_add(0, c6_index.flatten(), g.to(f64).flatten())
    virial = (v.to(f64)[..., :, None] * dv64[..., None, :]).sum(1)  # (B, 3, 3)
    return Accumulators(
        energy=acc.energy.index_add(0, t.system, (c6p * g).to(f64).sum(1)),
        grad_pos=grad_pos,
        virial=acc.virial.index_add(0, t.system, virial),
        grad_c6=grad_c6,
    )


@cache
def _compiled(with_grad: bool):
    return torch.compile(gradient_step if with_grad else energy_step, fullgraph=True, dynamic=False)


def get_pair_step_fn(with_grad: bool, compile: bool):
    """``fn(acc, c6, pos, cells, numbers, local, tile_inputs, scalars) -> acc`` for a flat batch
    of S systems, with the per-atom tables of the padded rows (Np): positions (Np, 3), the cell
    of each row (Np, 3, 3; zeros without PBC), atomic numbers and the index of each atom within
    its system; ``c6`` the flattened per-system C6 blocks. Energy only (``acc`` (S,)) or with
    gradients (``acc`` an ``Accumulators`` with energy (S,), virial (S, 3, 3) and dE/dC6 like
    ``c6``)."""
    if compile:
        return _compiled(with_grad)
    return gradient_step if with_grad else energy_step
