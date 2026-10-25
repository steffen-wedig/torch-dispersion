"""Gaussian reference weights and CN-dependent pairwise C6 (simple-dftd3 ``d3_model``)."""

import torch
from torch import Tensor

WF = 4.0


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


def pairwise_c6(weights: Tensor, species: Tensor, c6_species: Tensor) -> Tensor:
    """C6_ij = sum_ab w_ia w_jb C6_ref(Z_i, a, Z_j, b).

    weights (N, R), species (N,) local species index, c6_species (S, S, R, R). Returns (N, N).
    Memory is O(N^2 R) rather than O(N^2 R^2).
    """
    # A[i, t, b] = sum_a w_ia C6_ref(s_i, a, t, b)
    a = torch.einsum("ia,itab->itb", weights, c6_species[species])
    return torch.einsum("ijb,jb->ij", a[:, species, :], weights)
