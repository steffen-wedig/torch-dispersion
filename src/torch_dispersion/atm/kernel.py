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
from typing import NamedTuple, Protocol

import torch
from torch import Tensor

from ..core.tiles import Accumulators, TileLayout


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


def triple_inputs(r0: Tensor):
    """``make_inputs`` of ``core.tiles.run_tiles`` (order 2) for the tile pair (p, q) of a
    block; ``r0`` is the (Z, Z) damping radius table."""
    diagonal = {flag: torch.tensor(flag, device=r0.device) for flag in (True, False)}

    def make(layout: TileLayout, kb: int, tiles: tuple[int, ...]) -> TileInputs:
        p, q = tiles
        return TileInputs(
            centre=layout.centres[kb],
            idx_n=layout.indices[kb, p],
            idx_m=layout.indices[kb, q],
            mask_n=layout.mask[kb, p],
            mask_m=layout.mask[kb, q],
            z_centre=layout.z_centres[kb],
            z_n=layout.z[kb, p],
            z_m=layout.z[kb, q],
            r0=r0,
            diagonal=diagonal[p == q],
        )

    return make


def _switch(r2: Tensor, cutoff: Tensor, width: Tensor) -> tuple[Tensor, Tensor]:
    """Quintic switch s(r) = x^3 (10 - 15 x + 6 x^2), x = (cutoff - r) / width, and ds/d(r^2).

    s = 1 and ds/d(r^2) = 0 when width = 0 (hard cutoff).
    """
    on = width > 0.0
    safe_width = torch.where(on, width, 1.0)
    r = torch.sqrt(r2)
    x = ((cutoff - r) / safe_width).clamp(0.0, 1.0)
    sw = torch.where(on, x * x * x * (10.0 + x * (-15.0 + 6.0 * x)), 1.0)
    dsw_dr = -30.0 * x * x * (1.0 - x) * (1.0 - x) / safe_width  # 0 outside (inner, cutoff)
    return sw, torch.where(on, 0.5 * dsw_dr / r, 0.0)


class _Terms(NamedTuple):
    """Per-triangle quantities of one tile. Leg quantities of the centre legs are (B, T) (they
    depend on n or m only), the rest (B, T, T); masked entries hold safe values."""

    valid: Tensor
    u_ij: Tensor  # (B, T) squared legs, 1 where invalid
    u_ik: Tensor  # (B, T)
    u_jk: Tensor
    inv_u_ij: Tensor  # (B, T)
    inv_u_ik: Tensor  # (B, T)
    inv_u_jk: Tensor
    k: Tensor  # s9 sqrt(|C6_ij C6_ik C6_jk|) / 3, 0 where a C6 is below eps
    tdmp: Tensor  # 6 (r0 / r1)^((alpha + 2) / 3)
    fdmp: Tensor  # 1 / (1 + tdmp)
    ang: Tensor
    inv_r3: Tensor  # 1 / (r_ij r_ik r_jk)^3
    inv_r5: Tensor
    sw: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor]]
    e: Tensor  # triangle energy including the 1/3, 0 where invalid


class TriangleScalars(Protocol):
    """The scalars ``triangles`` and ``leg_derivatives`` read (``KernelScalars`` and the
    flat kernel's ``FlatScalars`` both have them)."""

    @property
    def exponent(self) -> Tensor: ...  # (alpha + 2) / 3
    @property
    def cutoff(self) -> Tensor: ...
    @property
    def width(self) -> Tensor: ...


def triangles(
    v_n: Tensor,
    v_m: Tensor,
    mask_n: Tensor,
    mask_m: Tensor,
    diagonal: Tensor,
    r0: Tensor,
    k: Tensor,
    s: TriangleScalars,
) -> _Terms:
    """The geometry of one tile's triangles (i, n, m) from the centre legs v_n, v_m (B, T, 3),
    their masks, the damping radius product r0 (B, T, T) and the prefactor k = s9 sqrt(C6 C6
    C6) / 3 (B, T, T). Only rsqrt, one pow and one division per triangle."""
    dtype = v_n.dtype
    eps = torch.finfo(dtype).eps
    cutoff2 = s.cutoff * s.cutoff
    tile = v_n.shape[1]

    r2_ij = v_n.square().sum(-1)  # (B, T)
    r2_ik = v_m.square().sum(-1)
    v_jk = v_m[:, None, :, :] - v_n[:, :, None, :]  # (B, T, T, 3)
    r2_jk = v_jk.square().sum(-1)

    ok_ij = mask_n & (r2_ij <= cutoff2) & (r2_ij > eps)
    ok_ik = mask_m & (r2_ik <= cutoff2) & (r2_ik > eps)
    ok_jk = (r2_jk <= cutoff2) & (r2_jk > eps)
    order = torch.arange(tile, device=v_n.device)
    upper = order[:, None] < order[None, :]
    valid = ok_ij[:, :, None] & ok_ik[:, None, :] & ok_jk & (upper | ~diagonal)

    u_ij = torch.where(ok_ij, r2_ij, 1.0)
    u_ik = torch.where(ok_ik, r2_ik, 1.0)
    u_jk = torch.where(valid, r2_jk, 1.0)
    ri_ij, ri_ik, ri_jk = torch.rsqrt(u_ij), torch.rsqrt(u_ik), torch.rsqrt(u_jk)
    inv_r1 = ri_ij[:, :, None] * ri_ik[:, None, :] * ri_jk
    inv_r2 = inv_r1 * inv_r1
    inv_r3 = inv_r2 * inv_r1
    inv_r5 = inv_r3 * inv_r2

    tdmp = 6.0 * (r0 * inv_r1) ** s.exponent
    fdmp = 1.0 / (1.0 + tdmp)
    a, b = u_ij[:, :, None], u_ik[:, None, :]
    ang = 0.375 * (a + u_jk - b) * (a - u_jk + b) * (-a + u_jk + b) * inv_r5 + inv_r3
    sw = (
        _switch(u_ij, s.cutoff, s.width),
        _switch(u_ik, s.cutoff, s.width),
        _switch(u_jk, s.cutoff, s.width),
    )
    sw_all = sw[0][0][:, :, None] * sw[1][0][:, None, :] * sw[2][0]
    e = torch.where(valid, k * fdmp * ang * sw_all, 0.0)
    return _Terms(
        valid=valid,
        u_ij=u_ij,
        u_ik=u_ik,
        u_jk=u_jk,
        inv_u_ij=ri_ij * ri_ij,
        inv_u_ik=ri_ik * ri_ik,
        inv_u_jk=ri_jk * ri_jk,
        k=k,
        tdmp=tdmp,
        fdmp=fdmp,
        ang=ang,
        inv_r3=inv_r3,
        inv_r5=inv_r5,
        sw=sw,
        e=e,
    )


def damping_radii(r0: Tensor, z_centre: Tensor, z_n: Tensor, z_m: Tensor) -> Tensor:
    """(B, T, T) product of the three sides' damping radii from the (Z, Z) table."""
    return (
        r0[z_centre[:, None], z_n][:, :, None]
        * r0[z_centre[:, None], z_m][:, None, :]
        * r0[z_n[:, :, None], z_m[:, None, :]]
    )


def _terms(
    v_n: Tensor, v_m: Tensor, c6: Tensor, t: TileInputs, s: KernelScalars
) -> tuple[_Terms, Tensor]:
    """Shared forward pass on one system's C6 (Np, Np): the triangle terms and 1/C6_jk (B, T, T;
    0 where C6 is below eps). The expensive float64 special functions
    are factorised: the centre legs are (B, T) work, 1/r1 needs one rsqrt per triangle, and
    sqrt(C6), 1/C6 are (Np, Np) tables gathered per triangle."""
    eps = torch.finfo(v_n.dtype).eps
    # C6 tables: triangles with any |C6| < eps get k = 0 (they are skipped in simple-dftd3)
    ok_c6 = c6.abs() >= eps
    c6_safe = torch.where(ok_c6, c6, 1.0)
    sqrt_c6 = torch.where(ok_c6, torch.sqrt(c6_safe.abs()), 0.0)
    inv_c6 = torch.where(ok_c6, 1.0 / c6_safe, 0.0)
    centre = t.centre[:, None]
    jk = (t.idx_n[:, :, None], t.idx_m[:, None, :])
    k = (s.s9 / 3.0) * (
        sqrt_c6[centre, t.idx_n][:, :, None] * sqrt_c6[centre, t.idx_m][:, None, :] * sqrt_c6[jk]
    )
    r0 = damping_radii(t.r0, t.z_centre, t.z_n, t.z_m)
    return triangles(v_n, v_m, t.mask_n, t.mask_m, t.diagonal, r0, k, s), inv_c6[jk]


def tile_energy(v_n: Tensor, v_m: Tensor, c6: Tensor, t: TileInputs, s: KernelScalars) -> Tensor:
    """ATM energy (Hartree, float64 scalar) of one tile. v_* (B, T, 3) bohr, c6 (Np, Np)."""
    return _terms(v_n, v_m, c6, t, s)[0].e.to(torch.float64).sum()


def tile_energy_and_grad_vjp(
    v_n: Tensor, v_m: Tensor, c6: Tensor, t: TileInputs, s: KernelScalars
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(energy, dE/dv_n, dE/dv_m, dE/dc6) via ``torch.func.vjp`` (the reference)."""

    def energy_fn(v_n: Tensor, v_m: Tensor, c6: Tensor) -> Tensor:
        return tile_energy(v_n, v_m, c6, t, s)

    out = torch.func.vjp(energy_fn, v_n, v_m, c6)
    energy, vjp_fn = out[0], out[1]
    g_n, g_m, g_c6 = vjp_fn(torch.ones_like(energy))
    return energy, g_n, g_m, g_c6


def leg_derivatives(x: _Terms, s: TriangleScalars, legs: tuple[int, ...]) -> list[Tensor]:
    """de/du_p (B, T, T) of the legs ``legs`` (0 = ij, 1 = ik, 2 = jk), u the squared lengths.

    With e = k fdmp(W) ang(u) sw(u), W = u_ij u_ik u_jk:

        d ang / du_p = 3/8 dN/du_p W^-5/2 - (15/16 N W^-5/2 + 3/2 W^-3/2) / u_p
        d fdmp / du_p = fdmp^2 tdmp (alpha + 2) / 6 / u_p

    (N is the product of the three law-of-cosines factors)."""
    a, b, c = x.u_ij[:, :, None], x.u_ik[:, None, :], x.u_jk
    l1, l2, l3 = a + c - b, a - c + b, -a + c + b
    dn = (
        l2 * l3 + l1 * l3 - l1 * l2,  # dN/du_ij
        -l2 * l3 + l1 * l3 + l1 * l2,  # dN/du_ik
        l2 * l3 - l1 * l3 + l1 * l2,  # dN/du_jk
    )
    common = 0.9375 * (l1 * l2 * l3) * x.inv_r5 + 1.5 * x.inv_r3
    # d fdmp / du_p = fdmp^2 tdmp (alpha + 2) / 6 / u_p, and (alpha + 2) / 6 = exponent / 2
    dfd = x.fdmp * x.fdmp * x.tdmp * (0.5 * s.exponent)
    (s1, d1), (s2, d2), (s3, d3) = x.sw
    s1, d1, s2, d2 = s1[:, :, None], d1[:, :, None], s2[:, None, :], d2[:, None, :]
    sw = s1 * s2 * s3
    sw_d = (d1 * s2 * s3, s1 * d2 * s3, s1 * s2 * d3)
    inv_u = (x.inv_u_ij[:, :, None], x.inv_u_ik[:, None, :], x.inv_u_jk)
    fa = x.fdmp * x.ang
    grads = []
    for p in legs:
        dang = 0.375 * dn[p] * x.inv_r5 - common * inv_u[p]
        de = x.k * (sw * ((dfd * x.ang) * inv_u[p] + x.fdmp * dang) + fa * sw_d[p])
        grads.append(torch.where(x.valid, de, 0.0))
    return grads


def _analytic_pointwise(
    v_n: Tensor, v_m: Tensor, c6: Tensor, t: TileInputs, s: KernelScalars
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Per-triangle (e, de/du_ij, de/du_ik, de/du_jk, de/dC6_jk), all (B, T, T); e ~
    sqrt(C6_ij C6_ik C6_jk) gives de/dC6_x = e / (2 C6_x)."""
    x, inv_c6_jk = _terms(v_n, v_m, c6, t, s)
    g_ij, g_ik, g_jk = leg_derivatives(x, s, (0, 1, 2))
    return x.e, g_ij, g_ik, g_jk, 0.5 * x.e * inv_c6_jk


def _analytic_reduce(
    v_n: Tensor,
    v_m: Tensor,
    c6: Tensor,
    t: TileInputs,
    e: Tensor,
    g_ij: Tensor,
    g_ik: Tensor,
    g_jk: Tensor,
    h_jk: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Contract the per-triangle derivatives into (energy, dE/dv_n, dE/dv_m, dE/dC6).

    r2_ij = |v_n|^2, r2_ik = |v_m|^2, r2_jk = |v_m - v_n|^2, so the vector gradients are row and
    column sums plus batched matmuls, e.g. sum_m G_nm (v_m - v_n) = (G @ v_m) - v_n sum_m G. The
    C6 gradient is accumulated with one-hot matmuls instead of (heavily contended) atomics.
    """
    g_n = 2.0 * v_n * (g_ij.sum(2) + g_jk.sum(2))[..., None] - 2.0 * torch.bmm(g_jk, v_m)
    g_jk_t = g_jk.transpose(1, 2)
    g_m = 2.0 * v_m * (g_ik.sum(1) + g_jk.sum(1))[..., None] - 2.0 * torch.bmm(g_jk_t, v_n)

    eps = torch.finfo(c6.dtype).eps
    ok_c6 = c6.abs() >= eps
    inv_c6 = torch.where(ok_c6, 1.0 / torch.where(ok_c6, c6, 1.0), 0.0)
    centre = t.centre[:, None]
    atoms = torch.arange(c6.shape[0], device=v_n.device)
    hot_c = (t.centre[:, None] == atoms).to(v_n.dtype)  # (B, Np)
    hot_n = (t.idx_n[..., None] == atoms).to(v_n.dtype)  # (B, T, Np)
    hot_m = (t.idx_m[..., None] == atoms).to(v_n.dtype)
    h_ij = 0.5 * e.sum(2) * inv_c6[centre, t.idx_n]  # (B, T)
    h_ik = 0.5 * e.sum(1) * inv_c6[centre, t.idx_m]
    row_c = torch.bmm(h_ij[:, None, :], hot_n)[:, 0] + torch.bmm(h_ik[:, None, :], hot_m)[:, 0]
    g_c6 = hot_c.T @ row_c + torch.bmm(torch.bmm(hot_n.transpose(1, 2), h_jk), hot_m).sum(0)
    return e.to(torch.float64).sum(), g_n, g_m, g_c6


def tile_energy_and_grad_analytic(
    v_n: Tensor, v_m: Tensor, c6: Tensor, t: TileInputs, s: KernelScalars
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(energy, dE/dv_n, dE/dv_m, dE/dc6) from hand-written derivatives."""
    return _analytic_reduce(v_n, v_m, c6, t, *_analytic_pointwise(v_n, v_m, c6, t, s))


GRADIENTS = {"analytic": tile_energy_and_grad_analytic, "vjp": tile_energy_and_grad_vjp}


def _accumulate(
    acc: Accumulators,
    v_n: Tensor,
    v_m: Tensor,
    t: TileInputs,
    energy: Tensor,
    g_n: Tensor,
    g_m: Tensor,
    g_c6: Tensor,
) -> Accumulators:
    """Scatter one tile's vector gradients to atoms (one-hot matmuls, no atomics) and add."""
    f64 = torch.float64
    atoms = torch.arange(acc.grad_pos.shape[0], device=v_n.device)
    hot_c = (t.centre[:, None] == atoms).to(f64)  # (B, Np)
    hot_n = (t.idx_n[..., None] == atoms).to(f64).flatten(0, 1)  # (B T, Np)
    hot_m = (t.idx_m[..., None] == atoms).to(f64).flatten(0, 1)
    g_n, g_m = g_n.to(f64), g_m.to(f64)
    grad_pos = (
        hot_n.T @ g_n.flatten(0, 1)
        + hot_m.T @ g_m.flatten(0, 1)
        - hot_c.T @ (g_n.sum(1) + g_m.sum(1))
    )
    virial = torch.einsum("btx,bty->xy", v_n.to(f64), g_n)
    virial = virial + torch.einsum("btx,bty->xy", v_m.to(f64), g_m)
    return Accumulators(
        energy=acc.energy + energy,
        grad_pos=acc.grad_pos + grad_pos,
        virial=acc.virial + virial,
        grad_c6=acc.grad_c6 + g_c6.to(f64),
    )


def energy_step(
    acc: Tensor, v_n: Tensor, v_m: Tensor, c6: Tensor, t: TileInputs, s: KernelScalars
) -> Tensor:
    return acc + tile_energy(v_n, v_m, c6, t, s)


def _vjp_step(
    acc: Accumulators, v_n: Tensor, v_m: Tensor, c6: Tensor, t: TileInputs, s: KernelScalars
) -> Accumulators:
    return _accumulate(acc, v_n, v_m, t, *tile_energy_and_grad_vjp(v_n, v_m, c6, t, s))


def _analytic_reduce_step(
    acc: Accumulators, v_n: Tensor, v_m: Tensor, c6: Tensor, t: TileInputs, *pointwise: Tensor
) -> Accumulators:
    return _accumulate(acc, v_n, v_m, t, *_analytic_reduce(v_n, v_m, c6, t, *pointwise))


def _analytic_step(
    acc: Accumulators, v_n: Tensor, v_m: Tensor, c6: Tensor, t: TileInputs, s: KernelScalars
) -> Accumulators:
    pointwise = _analytic_pointwise(v_n, v_m, c6, t, s)
    return _analytic_reduce_step(acc, v_n, v_m, c6, t, *pointwise)


STEPS = {"analytic": _analytic_step, "vjp": _vjp_step}


def _compile(fn):
    return torch.compile(fn, fullgraph=True, dynamic=False)


@cache
def _compiled_tile(with_grad: bool, gradient: str):
    if not with_grad:
        return _compile(tile_energy)
    if gradient == "vjp":
        return _compile(tile_energy_and_grad_vjp)
    pointwise, reduce = _compile(_analytic_pointwise), _compile(_analytic_reduce)

    def analytic(v_n: Tensor, v_m: Tensor, c6: Tensor, t: TileInputs, s: KernelScalars):
        return reduce(v_n, v_m, c6, t, *pointwise(v_n, v_m, c6, t, s))

    return analytic


@cache
def _compiled_step(with_grad: bool, gradient: str):
    if not with_grad:
        return _compile(energy_step)
    if gradient == "vjp":
        return _compile(_vjp_step)
    # Two graphs: inductor would otherwise fuse the heavy per-triangle math into the row/column
    # reductions (a register-starved persistent reduction, ~4x slower on an H200). Splitting
    # makes the math a pointwise kernel and materialises its (B, T, T) outputs once.
    pointwise, reduce = _compile(_analytic_pointwise), _compile(_analytic_reduce_step)

    def analytic(acc, v_n: Tensor, v_m: Tensor, c6: Tensor, t: TileInputs, s: KernelScalars):
        return reduce(acc, v_n, v_m, c6, t, *pointwise(v_n, v_m, c6, t, s))

    return analytic


def get_tile_fn(with_grad: bool, compile: bool, gradient: str = "analytic"):
    """``fn(v_n, v_m, c6, tile_inputs, scalars)`` -> energy or (energy, dE/dv_n, dE/dv_m,
    dE/dc6). Compiled graphs are shared process-wide."""
    if compile:
        return _compiled_tile(with_grad, gradient if with_grad else "")
    return GRADIENTS[gradient] if with_grad else tile_energy


def get_step_fn(with_grad: bool, compile: bool, gradient: str = "analytic"):
    """``fn(acc, v_n, v_m, c6, tile_inputs, scalars) -> acc`` (a tile plus its accumulation):
    energy only (``acc`` a 0-d tensor) or with gradients (``acc`` an ``Accumulators``)."""
    if compile:
        return _compiled_step(with_grad, gradient if with_grad else "")
    return STEPS[gradient] if with_grad else energy_step
