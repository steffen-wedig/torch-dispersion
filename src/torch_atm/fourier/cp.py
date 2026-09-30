"""Separable C9: symmetric CP decomposition of the D3 reference C9 tensor.

D3 defines C9_ijk = sqrt(C6_ij C6_jk C6_ki) with the CN-interpolated C6. Interpolating the
reference tensor instead, C9_ijk = sum_pqs w_ip w_jq w_ks C9ref_pqs with
C9ref_pqs = sqrt(C6ref_pq C6ref_qs C6ref_sp), agrees to ~1e-5 on liquids. A rank-R symmetric CP
fit of C9ref,

    C9ref_pqs ~ sum_r mu_r V_pr V_qr V_sr,

then gives C9_ijk ~ sum_r mu_r a_ir a_jr a_kr with a_ir = sum_p w_ip V_pr, the separable form
the dense trace needs. The fit is per species set (like FourierD3's C6 eigen-decomposition),
minimises the mean squared relative error (in a normalised form) in float64 on the CPU and
is cached.
"""

from dataclasses import dataclass
from functools import cache

import torch
from torch import Tensor

from ..data import MAX_REFERENCES, reference_data


@dataclass(frozen=True)
class CPFit:
    """``factors`` (Z_count, MAX_REFERENCES, R): V per (species, reference), zero for missing
    references; ``weights`` (R,): mu. ``max_rel_error`` is over the reference tensor."""

    numbers: tuple[int, ...]
    factors: Tensor
    weights: Tensor
    max_rel_error: float
    rms_rel_error: float


def reference_c9(numbers: tuple[int, ...]) -> tuple[Tensor, list[tuple[int, int]]]:
    """C9ref over the references of ``numbers`` (float64, CPU) and the (Z, reference) list."""
    data = reference_data(torch.device("cpu"), torch.float64)
    refs = [(z, a) for z in numbers for a in range(int(data.n_ref[z]))]
    zi = torch.tensor([z for z, _ in refs])
    ai = torch.tensor([a for _, a in refs])
    c6 = data.c6_ref[zi[:, None], zi[None, :], ai[:, None], ai[None, :]]
    c9 = torch.sqrt(c6[:, :, None] * c6[None, :, :] * c6.T[:, None, :])
    return c9, refs


def _fit(c9: Tensor, rank: int, max_iter: int, seed: int) -> tuple[Tensor, Tensor]:
    """Fit in normalised form: with d_p = sqrt(C6ref_pp), Y = C9ref / (d_p d_q d_s) lies in
    (0.85, 1], so its absolute error is the relative error of C9ref and the problem is well
    conditioned. Y ~ sum_r mu_r u_pr u_qr u_sr by LBFGS from u = 1 (the geometric-mean term)
    plus random terms; V = d u."""
    gen = torch.Generator().manual_seed(seed)
    d = torch.diagonal(torch.diagonal(c9)) ** (1.0 / 3.0)
    y = c9 / (d[:, None, None] * d[None, :, None] * d[None, None, :])
    p = c9.shape[0]
    u = torch.ones(p, rank, dtype=c9.dtype)
    u = u + 0.3 * torch.randn(p, rank, generator=gen, dtype=c9.dtype) * (torch.arange(rank) > 0)
    u = u.requires_grad_()
    mu = torch.full((rank,), 1.0 / rank, dtype=c9.dtype, requires_grad=True)
    opt = torch.optim.LBFGS(
        [u, mu],
        max_iter=max_iter,
        history_size=100,
        tolerance_grad=1e-16,
        tolerance_change=1e-20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = (torch.einsum("r,pr,qr,sr->pqs", mu, u, u, u) - y).square().mean()
        loss.backward()
        return loss

    opt.step(closure)
    return d[:, None] * u.detach(), mu.detach()


@cache
def fit_c9(
    numbers: tuple[int, ...],
    rank: int,
    max_iter: int = 300,
    restarts: int = 3,
    max_restarts: int = 8,
    target: float = 1e-2,
) -> CPFit:
    """Best of ``restarts`` fits (by max relative error), with more restarts (up to
    ``max_restarts``) while the best is worse than ``target`` (a start can stall in the rank-1
    solution). ``numbers``: sorted unique Z. About 0.3 s per fit for a few species."""
    c9, refs = reference_c9(numbers)
    best = None
    for seed in range(max_restarts):
        if seed >= restarts and best is not None and best[0] <= target:
            break
        v, mu = _fit(c9, rank, max_iter, seed)
        rel = torch.einsum("r,pr,qr,sr->pqs", mu, v, v, v) / c9 - 1.0
        err = float(rel.abs().max())
        if best is None or err < best[0]:
            best = (err, float(rel.square().mean().sqrt()), v, mu)
    assert best is not None
    err, rms, v, mu = best
    z_max = max(numbers)
    factors = torch.zeros(z_max + 1, MAX_REFERENCES, rank, dtype=torch.float64)
    for row, (z, a) in enumerate(refs):
        factors[z, a] = v[row]
    return CPFit(numbers, factors, mu, err, rms)


def atom_factors(weights: Tensor, numbers: Tensor, fit: CPFit) -> Tensor:
    """a_ir = sum_p w_ip V_(Z_i, p) r. weights (N, 7), numbers (N,) (0 for dummies, whose
    weights are zero). Returns (N, R) in the dtype and device of ``weights``."""
    table = fit.factors.to(device=weights.device, dtype=weights.dtype)
    return torch.einsum("ip,ipr->ir", weights, table[numbers])
