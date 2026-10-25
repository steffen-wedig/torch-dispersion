"""Pairwise C6 from per-atom reference weights (the D3 and D4 contraction)."""

import torch
from torch import Tensor


def pairwise_c6(weights: Tensor, species: Tensor, c6_species: Tensor) -> Tensor:
    """C6_ij = sum_ab w_ia w_jb C6_ref(Z_i, a, Z_j, b).

    weights (N, R), species (N,) local species index, c6_species (S, S, R, R). Returns (N, N).
    Memory is O(N^2 R) rather than O(N^2 R^2).
    """
    # A[i, t, b] = sum_a w_ia C6_ref(s_i, a, t, b)
    a = torch.einsum("ia,itab->itb", weights, c6_species[species])
    return torch.einsum("ijb,jb->ij", a[:, species, :], weights)
