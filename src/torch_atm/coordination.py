"""D3 coordination number (simple-dftd3 ``get_coordination_number``, exp counting)."""

import torch
from torch import Tensor

K1 = 16.0
CN_MIN_R2 = 1e-12


def coordination_number(
    vectors: Tensor, mask: Tensor, rcov_centre: Tensor, rcov_neighbor: Tensor, cutoff: float
) -> Tensor:
    """CN_i = sum_m 1 / (1 + exp(-k1 (Rcov_im / r_im - 1))) with a hard cutoff.

    vectors (N, M, 3) in bohr, mask (N, M), rcov_centre (N,), rcov_neighbor (N, M).
    Fixed-shape and branch-free, so it compiles with ``fullgraph=True``.
    """
    r2 = vectors.square().sum(-1)
    valid = mask & (r2 <= cutoff * cutoff) & (r2 >= CN_MIN_R2)
    r = torch.sqrt(torch.where(valid, r2, 1.0))
    rc = rcov_centre[:, None] + rcov_neighbor
    count = 1.0 / (1.0 + torch.exp(-K1 * (rc / r - 1.0)))
    return torch.where(valid, count, 0.0).sum(-1)


def smooth_coordination_number(
    vectors: Tensor, mask: Tensor, rcov_centre: Tensor, rcov_neighbor: Tensor, cutoff: float
) -> Tensor:
    """FourierD3's ``smooth_cut`` CN: the D3 counting function with a variable steepness that
    takes every contribution smoothly to zero at ``cutoff`` (FourierD3 ``compute_cn_smooth``).

    With Rc = Rcov_i + Rcov_j (the D3 radii, 4/3 already included) and the midpoint
    R_mid = (3/4 Rc + cutoff) / 2, the steepness is k(r) = 16 + (r - R_mid)^2 / ((cutoff - r)^2
    + 1e-6) beyond R_mid and 16 inside. Same arguments as ``coordination_number``.
    """
    r2 = vectors.square().sum(-1)
    valid = mask & (r2 <= cutoff * cutoff) & (r2 >= CN_MIN_R2)
    r = torch.sqrt(torch.where(valid, r2, 1.0))
    rc = rcov_centre[:, None] + rcov_neighbor
    r_mid = 0.5 * (0.75 * rc) + 0.5 * cutoff
    r_coeff = torch.maximum(r, r_mid)
    k = K1 + (r_coeff - r_mid).square() / ((cutoff - r_coeff).square() + 1e-6)
    count = torch.sigmoid(k * (rc / r - 1.0))
    return torch.where(valid, count, 0.0).sum(-1)
