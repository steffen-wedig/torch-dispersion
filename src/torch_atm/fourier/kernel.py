"""Short-range correction kernel: exact D3 minus the undamped separable term.

Per triangle with all legs within the short-range cutoff (switched by the quintic s),

    e = [k_D3 fdmp - k_CP] ang s_ij s_ik s_jk,
    k_D3 = s9/3 sqrt(C6_ij C6_ik C6_jk),   k_CP = s9/3 sum_r mu_r a_ir a_nr a_mr,

so that trace (all triangles, undamped, k_CP) + correction = D3 inside the cutoff. The tiling,
padding and counting are those of ``torch_atm.kernel``; this adds the (Np, R) factor table
``a`` as a second system-sized input and its gradient.
"""

from functools import cache
from typing import NamedTuple

import torch
from torch import Tensor

from ..kernel import (
    Accumulators,
    KernelScalars,
    TileInputs,
    _accumulate,
    _analytic_reduce,
    _terms,
)


class SRScalars(NamedTuple):
    base: KernelScalars
    mu: Tensor  # (R,)


class SRAccumulators(NamedTuple):
    energy: Tensor  # ()
    grad_pos: Tensor  # (Np, 3)
    virial: Tensor  # (3, 3)
    grad_c6: Tensor  # (Np, Np)
    grad_a: Tensor  # (Np, R)


def zero_sr_accumulators(n_padded: int, rank: int, device: torch.device) -> SRAccumulators:
    f64 = torch.float64
    return SRAccumulators(
        energy=torch.zeros((), dtype=f64, device=device),
        grad_pos=torch.zeros((n_padded, 3), dtype=f64, device=device),
        virial=torch.zeros((3, 3), dtype=f64, device=device),
        grad_c6=torch.zeros((n_padded, n_padded), dtype=f64, device=device),
        grad_a=torch.zeros((n_padded, rank), dtype=f64, device=device),
    )


def _cp_prefactor(a: Tensor, t: TileInputs, mu: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(sum_r mu_r a_c a_n a_m (B, T, T), mu a_c (B, R), a_n (B, T, R), a_m (B, T, R))."""
    a_c = a[t.centre] * mu  # (B, R)
    a_n, a_m = a[t.idx_n], a[t.idx_m]
    k = torch.bmm(a_n * a_c[:, None, :], a_m.transpose(1, 2))
    return k, a_c, a_n, a_m


def _sw_all(x) -> Tensor:
    return x.sw[0][0][:, :, None] * x.sw[1][0][:, None, :] * x.sw[2][0]


def sr_tile_energy(
    v_n: Tensor, v_m: Tensor, c6: Tensor, a: Tensor, t: TileInputs, s: SRScalars
) -> Tensor:
    x = _terms(v_n, v_m, c6, t, s.base)
    k_cp = _cp_prefactor(a, t, s.mu)[0]
    e_cp = torch.where(x.valid, (s.base.s9 / 3.0) * k_cp * x.ang * _sw_all(x), 0.0)
    return (x.e - e_cp).to(torch.float64).sum()


def sr_tile_energy_and_grad_vjp(
    v_n: Tensor, v_m: Tensor, c6: Tensor, a: Tensor, t: TileInputs, s: SRScalars
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """(energy, dE/dv_n, dE/dv_m, dE/dc6, dE/da) via ``torch.func.vjp``."""

    def energy_fn(v_n: Tensor, v_m: Tensor, c6: Tensor, a: Tensor) -> Tensor:
        return sr_tile_energy(v_n, v_m, c6, a, t, s)

    out = torch.func.vjp(energy_fn, v_n, v_m, c6, a)
    energy, vjp_fn = out[0], out[1]
    g_n, g_m, g_c6, g_a = vjp_fn(torch.ones_like(energy))
    return energy, g_n, g_m, g_c6, g_a


def _sr_pointwise(
    v_n: Tensor, v_m: Tensor, c6: Tensor, a: Tensor, t: TileInputs, s: SRScalars
) -> tuple[Tensor, ...]:
    """Per-triangle (e_D3, e_CP, de/du_ij, de/du_ik, de/du_jk, de_D3/dC6_jk, q), (B, T, T).

    As ``kernel._analytic_pointwise`` with the prefactor of ang s replaced by
    K = k_D3 fdmp - k_CP (the damping derivative keeps k_D3); q = s9/3 ang s is the
    derivative of e_CP with respect to sum_r mu_r a a a.
    """
    x = _terms(v_n, v_m, c6, t, s.base)
    k_raw = _cp_prefactor(a, t, s.mu)[0]
    k_cp = (s.base.s9 / 3.0) * k_raw
    a_, b_, c_ = x.u_ij[:, :, None], x.u_ik[:, None, :], x.u_jk
    l1, l2, l3 = a_ + c_ - b_, a_ - c_ + b_, -a_ + c_ + b_
    dn = (
        l2 * l3 + l1 * l3 - l1 * l2,
        -l2 * l3 + l1 * l3 + l1 * l2,
        l2 * l3 - l1 * l3 + l1 * l2,
    )
    common = 0.9375 * (l1 * l2 * l3) * x.inv_r5 + 1.5 * x.inv_r3
    dfd = x.fdmp * x.fdmp * x.tdmp * (0.5 * s.base.exponent)
    (s1, d1), (s2, d2), (s3, d3) = x.sw
    s1, d1, s2, d2 = s1[:, :, None], d1[:, :, None], s2[:, None, :], d2[:, None, :]
    sw = s1 * s2 * s3
    sw_d = (d1 * s2 * s3, s1 * d2 * s3, s1 * s2 * d3)
    inv_u = (x.inv_u_ij[:, :, None], x.inv_u_ik[:, None, :], x.inv_u_jk)
    pref = x.k * x.fdmp - k_cp
    grads = []
    for p in range(3):
        dang = 0.375 * dn[p] * x.inv_r5 - common * inv_u[p]
        de = sw * (x.k * dfd * x.ang * inv_u[p] + pref * dang) + pref * x.ang * sw_d[p]
        grads.append(torch.where(x.valid, de, 0.0))
    q = torch.where(x.valid, (s.base.s9 / 3.0) * x.ang * sw, 0.0)
    e_cp = k_raw * q
    return x.e, e_cp, grads[0], grads[1], grads[2], 0.5 * x.e * x.inv_c6_jk, q


def _sr_reduce(
    v_n: Tensor,
    v_m: Tensor,
    c6: Tensor,
    a: Tensor,
    t: TileInputs,
    mu: Tensor,
    e_d3: Tensor,
    e_cp: Tensor,
    g_ij: Tensor,
    g_ik: Tensor,
    g_jk: Tensor,
    h_jk: Tensor,
    q: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """(energy, dE/dv_n, dE/dv_m, dE/dC6, dE/da). The C6 gradient comes from the D3 part only
    (``kernel._analytic_reduce`` with e_D3); the a gradient contracts q with the factors."""
    e_d3_sum, g_n, g_m, g_c6 = _analytic_reduce(v_n, v_m, c6, t, e_d3, g_ij, g_ik, g_jk, h_jk)
    energy = e_d3_sum - e_cp.to(torch.float64).sum()
    a_c = a[t.centre]
    a_n, a_m = a[t.idx_n], a[t.idx_m]
    q_am = torch.bmm(q, a_m)  # (B, T, R): sum_m q_nm a_m
    q_an = torch.bmm(q.transpose(1, 2), a_n)  # (B, T, R): sum_n q_nm a_n
    g_c = -(mu * (a_n * q_am).sum(1))  # (B, R)
    g_an = -(mu * a_c)[:, None, :] * q_am
    g_am = -(mu * a_c)[:, None, :] * q_an
    atoms = torch.arange(a.shape[0], device=a.device)
    hot_c = (t.centre[:, None] == atoms).to(a.dtype)
    hot_n = (t.idx_n[..., None] == atoms).to(a.dtype).flatten(0, 1)
    hot_m = (t.idx_m[..., None] == atoms).to(a.dtype).flatten(0, 1)
    g_a = hot_c.T @ g_c + hot_n.T @ g_an.flatten(0, 1) + hot_m.T @ g_am.flatten(0, 1)
    return energy, g_n, g_m, g_c6, g_a


def _sr_accumulate(
    acc: SRAccumulators,
    v_n: Tensor,
    v_m: Tensor,
    t: TileInputs,
    energy: Tensor,
    g_n: Tensor,
    g_m: Tensor,
    g_c6: Tensor,
    g_a: Tensor,
) -> SRAccumulators:
    base = _accumulate(Accumulators(*acc[:4]), v_n, v_m, t, energy, g_n, g_m, g_c6)
    return SRAccumulators(*base, grad_a=acc.grad_a + g_a.to(torch.float64))


def sr_energy_step(
    acc: Tensor, v_n: Tensor, v_m: Tensor, c6: Tensor, a: Tensor, t: TileInputs, s: SRScalars
) -> Tensor:
    return acc + sr_tile_energy(v_n, v_m, c6, a, t, s)


def _sr_vjp_step(
    acc: SRAccumulators,
    v_n: Tensor,
    v_m: Tensor,
    c6: Tensor,
    a: Tensor,
    t: TileInputs,
    s: SRScalars,
) -> SRAccumulators:
    return _sr_accumulate(acc, v_n, v_m, t, *sr_tile_energy_and_grad_vjp(v_n, v_m, c6, a, t, s))


def _sr_reduce_step(
    acc: SRAccumulators,
    v_n: Tensor,
    v_m: Tensor,
    c6: Tensor,
    a: Tensor,
    t: TileInputs,
    mu: Tensor,
    *pointwise: Tensor,
) -> SRAccumulators:
    return _sr_accumulate(acc, v_n, v_m, t, *_sr_reduce(v_n, v_m, c6, a, t, mu, *pointwise))


def _sr_analytic_step(
    acc: SRAccumulators,
    v_n: Tensor,
    v_m: Tensor,
    c6: Tensor,
    a: Tensor,
    t: TileInputs,
    s: SRScalars,
) -> SRAccumulators:
    pointwise = _sr_pointwise(v_n, v_m, c6, a, t, s)
    return _sr_reduce_step(acc, v_n, v_m, c6, a, t, s.mu, *pointwise)


SR_STEPS = {"analytic": _sr_analytic_step, "vjp": _sr_vjp_step}


def _compile(fn):
    return torch.compile(fn, fullgraph=True, dynamic=False)


@cache
def _compiled_sr_step(with_grad: bool, gradient: str):
    if not with_grad:
        return _compile(sr_energy_step)
    if gradient == "vjp":
        return _compile(_sr_vjp_step)
    # two graphs for the same reason as ``kernel._compiled_step``
    pointwise, reduce = _compile(_sr_pointwise), _compile(_sr_reduce_step)

    def analytic(acc, v_n, v_m, c6, a, t: TileInputs, s: SRScalars):
        return reduce(acc, v_n, v_m, c6, a, t, s.mu, *pointwise(v_n, v_m, c6, a, t, s))

    return analytic


def get_sr_step_fn(with_grad: bool, compile: bool, gradient: str = "analytic"):
    """``fn(acc, v_n, v_m, c6, a, tile_inputs, scalars) -> acc``."""
    if compile:
        return _compiled_sr_step(with_grad, gradient if with_grad else "")
    return SR_STEPS[gradient] if with_grad else sr_energy_step
