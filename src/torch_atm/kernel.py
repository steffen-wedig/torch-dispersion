"""Fixed-shape ATM triple kernel.

One call evaluates B centres i against one tile pair (p, q) of their neighbour lists: sites n
from tile p and m from tile q, each tile ``T`` entries wide. With q >= p and, on diagonal tiles
(p == q), only n < m, every unordered pair {n, m} of N(i) is visited once and

    E = (1/3) sum_i sum_{n < m} e(i, n, m).

The heavy tensors are (B, T) and (B, T, T), with B and T fixed by the settings, so the compiled
graph does not depend on the system. The only system-sized input is the C6 matrix (Np, Np), whose
two dimensions the caller marks dynamic. Parameters are 0-d tensors, so one graph serves every
parameter set (dynamo guards on Python floats would recompile per cutoff). The kernel is a pure
function without data-dependent control flow: masked entries are neutralised with ``torch.where``
and safe values before any division, sqrt or power.

Nomenclature follows simple-dftd3: ij = (i, n), ik = (i, m), jk = (n, m).
"""

from dataclasses import dataclass
from functools import cache
from typing import NamedTuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class KernelParams:
    s9: float
    alpha: float  # damping-set alp; the ATM exponent is alpha + 2
    cutoff: float  # bohr
    width: float  # bohr, 0 = hard cutoff

    def scalars(self, dtype: torch.dtype, device: torch.device) -> "KernelScalars":
        def t(x: float) -> Tensor:
            return torch.tensor(x, dtype=dtype, device=device)

        return KernelScalars(
            s9=t(self.s9),
            exponent=t((self.alpha + 2.0) / 3.0),
            cutoff=t(self.cutoff),
            width=t(self.width),
        )


class KernelScalars(NamedTuple):
    s9: Tensor
    exponent: Tensor  # (alpha + 2) / 3
    cutoff: Tensor
    width: Tensor  # 0 disables the switching function


class TileInputs(NamedTuple):
    """Everything except the differentiable inputs (v_n, v_m, c6)."""

    centre: Tensor  # (B,) atom index of the centres (dummy index for padded centres)
    idx_n: Tensor  # (B, T) atom index of the n sites
    idx_m: Tensor  # (B, T)
    mask_n: Tensor  # (B, T) bool
    mask_m: Tensor  # (B, T) bool
    z_centre: Tensor  # (B,) atomic numbers (0 for dummies)
    z_n: Tensor  # (B, T)
    z_m: Tensor  # (B, T)
    r0: Tensor  # (Z, Z) rs9 * R0 table, bohr
    diagonal: Tensor  # () bool: p == q, keep only n < m


def _switch(r2: Tensor, cutoff: Tensor, width: Tensor) -> Tensor:
    """Quintic switch s(r) = x^3 (10 - 15 x + 6 x^2), x = (cutoff - r) / width; 1 if width = 0."""
    on = width > 0.0
    x = ((cutoff - torch.sqrt(r2)) / torch.where(on, width, 1.0)).clamp(0.0, 1.0)
    return torch.where(on, x * x * x * (10.0 + x * (-15.0 + 6.0 * x)), 1.0)


def tile_energy(v_n: Tensor, v_m: Tensor, c6: Tensor, t: TileInputs, s: KernelScalars) -> Tensor:
    """ATM energy (Hartree, float64 scalar) of one tile. v_* (B, T, 3) bohr, c6 (Np, Np)."""
    dtype = v_n.dtype
    eps = torch.finfo(dtype).eps
    cutoff2 = s.cutoff * s.cutoff
    tile = v_n.shape[1]

    r2_ij = v_n.square().sum(-1)  # (B, T)
    r2_ik = v_m.square().sum(-1)
    v_jk = v_m[:, None, :, :] - v_n[:, :, None, :]  # (B, T, T, 3)
    r2_jk = v_jk.square().sum(-1)

    ok_ij = t.mask_n & (r2_ij <= cutoff2) & (r2_ij > eps)
    ok_ik = t.mask_m & (r2_ik <= cutoff2) & (r2_ik > eps)
    ok_jk = (r2_jk <= cutoff2) & (r2_jk > eps)
    order = torch.arange(tile, device=v_n.device)
    upper = order[:, None] < order[None, :]
    valid = ok_ij[:, :, None] & ok_ik[:, None, :] & ok_jk & (upper | ~t.diagonal)

    c6_ij = c6[t.centre[:, None], t.idx_n]  # (B, T)
    c6_ik = c6[t.centre[:, None], t.idx_m]
    c6_jk = c6[t.idx_n[:, :, None], t.idx_m[:, None, :]]  # (B, T, T)
    c6_prod = c6_ij[:, :, None] * c6_ik[:, None, :] * c6_jk
    valid = valid & (c6_ij.abs() >= eps)[:, :, None] & (c6_ik.abs() >= eps)[:, None, :]
    valid = valid & (c6_jk.abs() >= eps)

    one = torch.ones((), dtype=dtype, device=v_n.device)
    r2_ij = torch.where(valid, r2_ij[:, :, None], one)
    r2_ik = torch.where(valid, r2_ik[:, None, :], one)
    r2_jk = torch.where(valid, r2_jk, one)
    c9 = s.s9 * torch.sqrt(torch.where(valid, c6_prod, one).abs())

    r0 = (
        t.r0[t.z_centre[:, None], t.z_n][:, :, None]
        * t.r0[t.z_centre[:, None], t.z_m][:, None, :]
        * t.r0[t.z_n[:, :, None], t.z_m[:, None, :]]
    )
    r2 = r2_ij * r2_ik * r2_jk
    r1 = torch.sqrt(r2)
    r3 = r2 * r1
    r5 = r3 * r2
    fdmp = 1.0 / (1.0 + 6.0 * (r0 / r1) ** s.exponent)
    ang = (
        0.375 * (r2_ij + r2_jk - r2_ik) * (r2_ij - r2_jk + r2_ik) * (-r2_ij + r2_jk + r2_ik) / r5
        + 1.0 / r3
    )
    sw = (
        _switch(r2_ij, s.cutoff, s.width)
        * _switch(r2_ik, s.cutoff, s.width)
        * _switch(r2_jk, s.cutoff, s.width)
    )
    e = torch.where(valid, c9 * fdmp * ang * sw, 0.0)
    return e.to(torch.float64).sum() / 3.0


def tile_energy_and_grad(
    v_n: Tensor, v_m: Tensor, c6: Tensor, t: TileInputs, s: KernelScalars
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(energy, dE/dv_n, dE/dv_m, dE/dc6) via ``torch.func.vjp``."""

    def energy_fn(v_n: Tensor, v_m: Tensor, c6: Tensor) -> Tensor:
        return tile_energy(v_n, v_m, c6, t, s)

    out = torch.func.vjp(energy_fn, v_n, v_m, c6)
    energy, vjp_fn = out[0], out[1]
    g_n, g_m, g_c6 = vjp_fn(torch.ones_like(energy))
    return energy, g_n, g_m, g_c6


@cache
def _compiled(with_grad: bool):
    fn = tile_energy_and_grad if with_grad else tile_energy
    return torch.compile(fn, fullgraph=True, dynamic=False)


def get_tile_fn(with_grad: bool, compile: bool):
    """``fn(v_n, v_m, c6, tile_inputs, scalars)``; compiled graphs are shared process-wide."""
    if compile:
        return _compiled(with_grad)
    return tile_energy_and_grad if with_grad else tile_energy
