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
