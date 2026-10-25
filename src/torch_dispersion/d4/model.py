"""D4 coordination number, charge-scaled reference weights and C6 (dftd4 ``d4_model``).

All functions are fixed-shape and branch-free, so they compile with ``fullgraph=True``.
"""

import math

import torch
from torch import Tensor

from ..coordination import CN_MIN_R2

KCN = 7.5
# electronegativity scaling of the D4 CN: k4 exp(-(|EN_i - EN_j| + k5)^2 / k6)
K4, K5, K6 = 4.10451, 19.08857, 2.0 * 11.28174**2
GA, GC, WF = 3.0, 2.0, 6.0  # charge scaling height and steepness, Gaussian weighting factor
MAX_GAUSSIANS = 3


def erf_count(r: Tensor, rc: Tensor, kcn: float) -> Tensor:
    """0.5 (1 + erf(-kcn (r - Rc) / Rc))."""
    return 0.5 * (1.0 + torch.erf(-kcn * (r - rc) / rc))


def d4_coordination_number(
    vectors: Tensor,
    mask: Tensor,
    rcov_centre: Tensor,
    rcov_neighbor: Tensor,
    en_centre: Tensor,
    en_neighbor: Tensor,
    cutoff: float,
) -> Tensor:
    """CN_i = sum_m den_im 0.5 (1 + erf(-7.5 (r_im - Rc) / Rc)), Rc = Rcov_i + Rcov_m, with the
    electronegativity factor den_im = k4 exp(-(|EN_i - EN_m| + k5)^2 / k6) and a hard cutoff.

    vectors (N, M, 3) bohr, mask (N, M), centre tables (N,), neighbour tables (N, M).
    """
    r2 = vectors.square().sum(-1)
    valid = mask & (r2 <= cutoff * cutoff) & (r2 >= CN_MIN_R2)
    r = torch.sqrt(torch.where(valid, r2, 1.0))
    rc = rcov_centre[:, None] + rcov_neighbor
    den = K4 * torch.exp(-((en_centre[:, None] - en_neighbor).abs() + K5).square() / K6)
    return torch.where(valid, den * erf_count(r, rc, KCN), 0.0).sum(-1)


def zeta(gi: Tensor, qref: Tensor, qmod: Tensor) -> Tensor:
    """Charge scaling exp(ga (1 - exp(gi (1 - qref / qmod)))), exp(ga) for qmod <= 0."""
    positive = qmod > 0.0
    safe = torch.where(positive, qmod, 1.0)
    scaled = torch.exp(GA * (1.0 - torch.exp(gi * (1.0 - qref / safe))))
    return torch.where(positive, scaled, math.exp(GA))


def reference_weights(
    cn: Tensor,
    q: Tensor,
    reference_cn: Tensor,
    reference_q: Tensor,
    n_gaussians: Tensor,
    zeff: Tensor,
    hardness: Tensor,
) -> Tensor:
    """Charge-scaled Gaussian reference weights zeta_ia(q_i) w_ia(CN_i), (N, R).

    w_ia = sum_{g <= ngw_a} exp(-g wf (CN_i - CN_a)^2) / sum_b sum_{g <= ngw_b} (...), with the
    dftd4 fallback where the normalisation underflows (below sqrt(tiny)) or the weight is not
    finite: one-hot on the references with the largest CN_ref, with zero derivative.
    cn, q, zeff, hardness (N,); reference_cn, reference_q, n_gaussians (N, R).
    """
    d2 = (cn[:, None] - reference_cn).square()
    g = torch.arange(1, MAX_GAUSSIANS + 1, device=cn.device, dtype=cn.dtype)
    gauss = torch.exp(-WF * g * d2[..., None])
    expw = torch.where(g <= n_gaussians[..., None], gauss, 0.0).sum(-1)
    norm = expw.sum(-1, keepdim=True)
    ok = norm > math.sqrt(torch.finfo(cn.dtype).tiny)
    weights = expw / torch.where(ok, norm, 1.0)
    exceptional = ~ok | ~torch.isfinite(weights)
    present = n_gaussians > 0
    masked_cn = torch.where(present, reference_cn, -torch.inf)
    largest = (masked_cn.amax(-1, keepdim=True) - reference_cn).abs() < 1e-12
    fallback = (largest & present).to(cn.dtype)
    weights = torch.where(exceptional, fallback, weights)
    z = zeff[:, None]
    scale = zeta((GC * hardness)[:, None], reference_q + z, q[:, None] + z)
    return torch.where(present, weights * scale, 0.0)


def polarizabilities(weights: Tensor, alpha_static: Tensor) -> Tensor:
    """Static atomic polarisabilities sum_a w_ia alpha_ia(0); alpha_static (N, R)."""
    return (weights * alpha_static).sum(-1)
