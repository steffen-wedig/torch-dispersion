"""Fixed-shape two-body kernel with rational (Becke-Johnson) damping, as in DFT-D3(BJ) and D4.

The kernel is independent of the C6 model: it takes C6 (Np, Np) and the (Z, Z) tables of
``rational_tables``. One call evaluates B centres i against one T-wide tile of their neighbour
lists (full list, both i -> j and j -> i, so every pair carries 1/2):

    E = -1/2 sum_{i, (j, T)} C6_ij [s6 / (r^6 + R0^6) + s8 rr_ij / (r^8 + R0^8)],
    rr_ij = 3 r4r2_i r4r2_j,  R0 = a1 sqrt(rr_ij) + a2,

for eps <= r^2 <= cutoff^2 (dftd4 and simple-dftd3 ``rational.f90``). The centres may belong to
different systems of a flat batch: C6 is a flat table (the per-system C6 blocks) read through a
per-pair index, and energy and virial are accumulated per system. Like the ATM kernel, the
heavy tensors are (B, T) with B and T fixed by the settings, the C6 table and the accumulators
are the only system-sized inputs (marked dynamic), and parameters are 0-d tensors. Gradients
are analytic: with u = r^2, de/du = -1/2 C6 (-3 s6 u^2 t6^2 - 4 s8 rr u^3 t8^2), and dE/dC6_ij
is the geometric factor itself.
"""

from functools import cache
from typing import NamedTuple

import torch
from torch import Tensor

from ..core.tiles import Accumulators, TileLayout


class PairScalars(NamedTuple):
    s6: Tensor
    s8: Tensor
    cutoff: Tensor


class PairTileInputs(NamedTuple):
    centre: Tensor  # (B,) atom index of the centres
    idx: Tensor  # (B, T) atom index of the neighbours
    mask: Tensor  # (B, T) bool
    z_centre: Tensor  # (B,)
    z: Tensor  # (B, T)
    c6_index: Tensor  # (B, T) entry of each pair in the flat C6 table
    system: Tensor  # (B,) system of each centre (energy and virial are per system)
    rr: Tensor  # (Z, Z) 3 r4r2_i r4r2_j
    r0: Tensor  # (Z, Z) a1 sqrt(rr) + a2


def rational_tables(r4r2: Tensor, a1: float, a2: float) -> tuple[Tensor, Tensor]:
    """(rr, R0) (Z, Z) from the r4r2 table: rr = 3 r4r2_i r4r2_j, R0 = a1 sqrt(rr) + a2."""
    rr = 3.0 * r4r2[:, None] * r4r2[None, :]
    return rr, a1 * torch.sqrt(rr) + a2


def pair_inputs(rr: Tensor, r0: Tensor, c6_index: Tensor, system: Tensor):
    """``make_inputs`` of ``core.tiles.run_tiles`` (order 1) for tile p of a block;
    ``c6_index`` and ``system`` are already tiled (``layout.tiled``, ``layout.rows``)."""

    def make(layout: TileLayout, kb: int, tiles: tuple[int, ...]) -> PairTileInputs:
        (p,) = tiles
        return PairTileInputs(
            centre=layout.centres[kb],
            idx=layout.indices[kb, p],
            mask=layout.mask[kb, p],
            z_centre=layout.z_centres[kb],
            z=layout.z[kb, p],
            c6_index=c6_index[kb, p],
            system=system[kb],
            rr=rr,
            r0=r0,
        )

    return make


def pair_scalars(s6: float, s8: float, cutoff: float, dtype, device) -> PairScalars:
    def t(x: float) -> Tensor:
        return torch.tensor(x, dtype=dtype, device=device)

    return PairScalars(s6=t(s6), s8=t(s8), cutoff=t(cutoff))


def _geometry(v: Tensor, t: PairTileInputs, s: PairScalars) -> tuple[Tensor, Tensor]:
    """Per pair (g, dg/du) with E = sum C6 g; zero outside the cutoff and in padding."""
    eps = torch.finfo(v.dtype).eps
    u = v.square().sum(-1)
    valid = t.mask & (u <= s.cutoff * s.cutoff) & (u >= eps)
    u = torch.where(valid, u, 1.0)
    zc = t.z_centre[:, None]
    rr, r0 = t.rr[zc, t.z], t.r0[zc, t.z]
    r0_2 = r0 * r0
    r0_6 = r0_2 * r0_2 * r0_2
    u2 = u * u
    t6 = 1.0 / (u2 * u + r0_6)
    t8 = 1.0 / (u2 * u2 + r0_6 * r0_2)
    g = -0.5 * (s.s6 * t6 + s.s8 * rr * t8)
    dg = 0.5 * (3.0 * s.s6 * u2 * t6 * t6 + 4.0 * s.s8 * rr * u2 * u * t8 * t8)
    return torch.where(valid, g, 0.0), torch.where(valid, dg, 0.0)


def energy_step(acc: Tensor, v: Tensor, c6: Tensor, t: PairTileInputs, s: PairScalars) -> Tensor:
    g, _ = _geometry(v, t, s)
    return acc.index_add(0, t.system, (c6[t.c6_index] * g).to(torch.float64).sum(1))


def gradient_step(
    acc: Accumulators, v: Tensor, c6: Tensor, t: PairTileInputs, s: PairScalars
) -> Accumulators:
    """Tile energy and gradients, scatter-added (pairs have little index contention)."""
    f64 = torch.float64
    g, dg = _geometry(v, t, s)
    c6p = c6[t.c6_index]
    dv = (2.0 * c6p * dg)[..., None] * v  # dE/dv, (B, T, 3)
    dv64 = dv.to(f64)
    grad_pos = acc.grad_pos.index_add(0, t.idx.flatten(), dv64.flatten(0, 1))
    grad_pos = grad_pos.index_add(0, t.centre, -dv64.sum(1))
    grad_c6 = acc.grad_c6.index_add(0, t.c6_index.flatten(), g.to(f64).flatten())
    virial = torch.einsum("btx,bty->bxy", v.to(f64), dv64)
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
    """``fn(acc, v, c6, tile_inputs, scalars) -> acc`` for a flat batch of S systems: ``c6`` is
    the flat C6 table read through ``c6_index``; energy only (``acc`` (S,)) or with gradients
    (``acc`` an ``Accumulators`` with energy (S,), virial (S, 3, 3) and dE/dC6 like ``c6``)."""
    if compile:
        return _compiled(with_grad)
    return gradient_step if with_grad else energy_step
