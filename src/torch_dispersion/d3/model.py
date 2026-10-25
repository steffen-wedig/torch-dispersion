"""D3 coordination number and CN-dependent C6 (simple-dftd3 ``d3_model``).

All functions are fixed-shape and branch-free, so they compile with ``fullgraph=True``.
"""

from functools import partial

import torch
from torch import Tensor

from ..core.c6 import pairwise_c6
from ..core.coordination import coordination_number

K1 = 16.0
WF = 4.0


def exp_count(r: Tensor, rc: Tensor) -> Tensor:
    """The D3 counting function 1 / (1 + exp(-k1 (Rc / r - 1)))."""
    return 1.0 / (1.0 + torch.exp(-K1 * (rc / r - 1.0)))


def smooth_count(r: Tensor, rc: Tensor, cutoff: float) -> Tensor:
    """FourierD3's ``smooth_cut`` counting function (``compute_cn_smooth``): the D3 function
    with a variable steepness that takes every contribution smoothly to zero at ``cutoff``.

    With Rc = Rcov_i + Rcov_j (the D3 radii, 4/3 already included) and the midpoint
    R_mid = (3/4 Rc + cutoff) / 2, the steepness is k(r) = 16 + (r - R_mid)^2 / ((cutoff - r)^2
    + 1e-6) beyond R_mid and 16 inside.
    """
    r_mid = 0.5 * (0.75 * rc) + 0.5 * cutoff
    r_coeff = torch.maximum(r, r_mid)
    k = K1 + (r_coeff - r_mid).square() / ((cutoff - r_coeff).square() + 1e-6)
    return torch.sigmoid(k * (rc / r - 1.0))


def d3_coordination_number(
    vectors: Tensor,
    mask: Tensor,
    rcov_centre: Tensor,
    rcov_neighbor: Tensor,
    cutoff: float,
    smooth: bool = False,
) -> Tensor:
    """CN_i = sum_m count(r_im, Rcov_i + Rcov_m) with a hard cutoff: the D3 ``exp_count``
    (simple-dftd3 ``get_coordination_number``) or, with ``smooth``, ``smooth_count``.

    vectors (N, M, 3) in bohr, mask (N, M), rcov_centre (N,), rcov_neighbor (N, M).
    """
    count = partial(smooth_count, cutoff=cutoff) if smooth else exp_count
    return coordination_number(vectors, mask, rcov_centre, rcov_neighbor, cutoff, count)


def reference_weights(cn: Tensor, reference_cn: Tensor, reference_mask: Tensor) -> Tensor:
    """w_ia = exp(-wf (CN_i - CN_ref,a)^2) / sum_b exp(-wf (CN_i - CN_ref,b)^2).

    cn (N,), reference_cn (N, R), reference_mask (N, R) marks the existing references.
    Where the normalisation underflows (its reciprocal is not finite), the weight is one-hot on
    the reference with the largest CN_ref, with zero derivative, as in simple-dftd3.
    """
    gw = torch.where(reference_mask, torch.exp(-WF * (cn[:, None] - reference_cn).square()), 0.0)
    norm = gw.sum(-1, keepdim=True)
    exceptional = ~torch.isfinite(1.0 / norm)
    weights = gw * (1.0 / torch.where(exceptional, 1.0, norm))
    masked_cn = torch.where(reference_mask, reference_cn, -torch.inf)
    largest = masked_cn == masked_cn.amax(-1, keepdim=True)
    fallback = (largest & reference_mask).to(cn.dtype)
    return torch.where(exceptional, fallback, weights)


def cn_weights_c6(
    vectors: Tensor,
    mask: Tensor,
    indices: Tensor,
    rcov: Tensor,
    ref_cn: Tensor,
    ref_mask: Tensor,
    species: Tensor,
    c6_species: Tensor,
    cutoff: float,
    smooth: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """(CN (Np,), reference weights (Np, R), C6 (Np, Np)) of a padded system (``D3System``
    tables) from its CN neighbour matrix (``vectors``, ``mask``, ``indices``)."""
    cn = d3_coordination_number(vectors, mask, rcov, rcov[indices], cutoff, smooth)
    weights = reference_weights(cn, ref_cn, ref_mask)
    return cn, weights, pairwise_c6(weights, species, c6_species)
