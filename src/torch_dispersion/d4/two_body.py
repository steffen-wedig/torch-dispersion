"""Fixed-shape two-body kernel of D4 with rational (Becke-Johnson) damping.

One call evaluates B centres i against one T-wide tile of their neighbour lists (full list, both
i -> j and j -> i, so every pair carries 1/2):

    E = -1/2 sum_{i, (j, T)} C6_ij [s6 / (r^6 + R0^6) + s8 rr_ij / (r^8 + R0^8)],
    rr_ij = 3 r4r2_i r4r2_j,  R0 = a1 sqrt(rr_ij) + a2,

for eps <= r^2 <= cutoff^2 (dftd4 ``rational.f90``). Like the ATM kernel, the heavy tensors are
(B, T) with B and T fixed by the settings, C6 (Np, Np) and the accumulators are the only
system-sized inputs (marked dynamic), and parameters are 0-d tensors. Gradients are analytic:
with u = r^2, de/du = -1/2 C6 (-3 s6 u^2 t6^2 - 4 s8 rr u^3 t8^2), and dE/dC6_ij is the
geometric factor itself.
"""

from functools import cache
from typing import NamedTuple

import torch
from torch import Tensor

from ..kernel import Accumulators


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
    rr: Tensor  # (Z, Z) 3 r4r2_i r4r2_j
    r0: Tensor  # (Z, Z) a1 sqrt(rr) + a2


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
    return acc + (c6[t.centre[:, None], t.idx] * g).to(torch.float64).sum()


def gradient_step(
    acc: Accumulators, v: Tensor, c6: Tensor, t: PairTileInputs, s: PairScalars
) -> Accumulators:
    """Tile energy and gradients, scatter-added (pairs have little index contention)."""
    f64 = torch.float64
    g, dg = _geometry(v, t, s)
    c6p = c6[t.centre[:, None], t.idx]
    dv = (2.0 * c6p * dg)[..., None] * v  # dE/dv, (B, T, 3)
    dv64 = dv.to(f64)
    grad_pos = acc.grad_pos.index_add(0, t.idx.flatten(), dv64.flatten(0, 1))
    grad_pos = grad_pos.index_add(0, t.centre, -dv64.sum(1))
    n_padded = c6.shape[0]
    flat = (t.centre[:, None] * n_padded + t.idx).flatten()
    grad_c6 = acc.grad_c6.flatten().index_add(0, flat, g.to(f64).flatten())
    return Accumulators(
        energy=acc.energy + (c6p * g).to(f64).sum(),
        grad_pos=grad_pos,
        virial=acc.virial + torch.einsum("btx,bty->xy", v.to(f64), dv64),
        grad_c6=grad_c6.reshape(n_padded, n_padded),
    )


@cache
def _compiled(with_grad: bool):
    return torch.compile(gradient_step if with_grad else energy_step, fullgraph=True, dynamic=False)


def get_pair_step_fn(with_grad: bool, compile: bool):
    """``fn(acc, v, c6, tile_inputs, scalars) -> acc``; energy only (``acc`` 0-d) or with
    gradients (``acc`` an ``Accumulators``)."""
    if compile:
        return _compiled(with_grad)
    return gradient_step if with_grad else energy_step
