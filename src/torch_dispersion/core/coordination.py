"""Coordination numbers over a padded neighbour matrix.

Every model's CN is a masked sum of a counting function of the distance (and of Rc, the sum of
the two covalent radii), optionally times a pair weight. The counting functions live with their
models: ``d3.model`` (exp and FourierD3's ``smooth_cut``), ``d4.model`` (erf, with the
electronegativity weight) and ``d4.eeq`` (erf, capped after the sum).
"""

from collections.abc import Callable

import torch
from torch import Tensor

# pairs closer than this (bohr^2) are coincident atoms: they do not count (and are rejected by
# the orchestration before any CN is taken)
CN_MIN_R2 = 1e-12


def coordination_number(
    vectors: Tensor,
    mask: Tensor,
    rcov_centre: Tensor,
    rcov_neighbor: Tensor,
    cutoff: float,
    count: Callable[[Tensor, Tensor], Tensor],
    weight: Tensor | None = None,
) -> Tensor:
    """CN_i = sum_m weight_im count(r_im, Rcov_i + Rcov_m) over CN_MIN_R2 <= r^2 <= cutoff^2.

    vectors (N, M, 3) in bohr, mask (N, M), rcov_centre (N,), rcov_neighbor (N, M), weight
    None or (N, M). Fixed-shape and branch-free, so it compiles with ``fullgraph=True``.
    """
    r2 = vectors.square().sum(-1)
    valid = mask & (r2 <= cutoff * cutoff) & (r2 >= CN_MIN_R2)
    r = torch.sqrt(torch.where(valid, r2, 1.0))
    c = count(r, rcov_centre[:, None] + rcov_neighbor)
    if weight is not None:
        c = weight * c
    return torch.where(valid, c, 0.0).sum(-1)
