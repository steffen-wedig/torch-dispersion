"""Dipole matrices and the undamped ATM energy as a dense trace.

With T(r) = (3 r r^T - r^2 1) / r^5 = grad grad (1/r), the ATM angular factor is

    Tr(T_ij T_jk T_ki) = -3 (1 + 3 cos a cos b cos c) / (r_ij r_jk r_ki)^3,

so for C9_ijk = sum_r mu_r a_ir a_jr a_kr the undamped ATM energy of all triangles is

    E = -(s9 / 18) sum_r mu_r < Tr[(D_r A(q))^3] >_q,     D_r = diag(a_r) (x) 1_3

(1/6 for the ordered vertices, 1/3 from the trace identity). A(q) is the 3N x 3N Bloch
dipole matrix A_ij(q) = sum_n T(x_j - x_i + n) e^{i q.n} without the (i = j, n = 0) term. The
average over a q-grid keeps closed triangles exactly; polygons that wind around the supercell
of the grid are the only error. Molecules use one real matrix (exact, no cutoff).

A(q) is evaluated by Ewald summation, 1/r = erfc(g r)/r + erf(g r)/r:

    real space: T_erfc(r) = C(r) r r^T - B(r) 1, over neighbour-list pairs (images included);
    reciprocal: -(4 pi / V) sum_{k = G - q} exp(-k^2 / 4 g^2) k k^T / k^2 e^{i k.(x_j - x_i)},
                one (3N x n_k) (n_k x 3N) matmul;
    self term:  + 4 g^3 / (3 sqrt(pi)) 1 on the diagonal blocks.

Grids are shifted off Gamma (q_f = (k + 1/2) / n), so k = G - q never vanishes and no
conditionally convergent term appears. Everything here runs in float64 (the undamped trace
is up to ~30x the damped energy, so float32 would lose digits).
"""

import math
from itertools import product

import torch
from torch import Tensor

from ...neighbors import NeighborMatrix


def q_grid(n: tuple[int, int, int], device: torch.device) -> tuple[Tensor, Tensor]:
    """Shifted Monkhorst-Pack grid in fractional coordinates, reduced by time reversal.

    q and -q give complex-conjugate matrices and equal real traces, so one of each pair is kept
    with weight 2 (self-conjugate points, q = -q mod G, weight 1). Returns (fractions (n_q, 3),
    weights (n_q,)) with the weights summing to 1.
    """
    keep, weights = [], []
    for k in product(*(range(m) for m in n)):
        partner = tuple(m - 1 - x for x, m in zip(k, n, strict=True))
        if k <= partner:
            keep.append([(x + 0.5) / m for x, m in zip(k, n, strict=True)])
            weights.append(1.0 if k == partner else 2.0)
    total = math.prod(n)
    f64 = torch.float64
    return (
        torch.tensor(keep, dtype=f64, device=device),
        torch.tensor(weights, dtype=f64, device=device) / total,
    )


def grid_for_cell(cell: Tensor, supercell_length: float) -> tuple[int, int, int]:
    """Smallest n >= 2 per lattice direction with n * (plane spacing) >= ``supercell_length``."""
    c = cell.detach().to(torch.float64).cpu()
    volume = torch.linalg.det(c).abs()
    spacing = [
        float(volume / torch.linalg.cross(c[(i + 1) % 3], c[(i + 2) % 3]).norm()) for i in range(3)
    ]
    # at least 2: the q -> 0 correction needs grid points around Gamma
    n = [max(2, math.ceil(supercell_length / s - 1e-9)) for s in spacing]
    return n[0], n[1], n[2]


def dipole_tensor(vec: Tensor) -> Tensor:
    """T(r) (..., 3, 3) for displacements (..., 3); the caller masks r = 0."""
    r2 = vec.square().sum(-1)
    inv_r2 = 1.0 / r2
    inv_r3 = inv_r2 * torch.sqrt(inv_r2)
    eye = torch.eye(3, dtype=vec.dtype, device=vec.device)
    outer = vec[..., :, None] * vec[..., None, :]
    return (3.0 * inv_r2[..., None, None] * outer - eye) * inv_r3[..., None, None]


def molecular_dipole_matrix(positions: Tensor) -> Tensor:
    """(3N, 3N) real dipole matrix of an isolated system."""
    n = positions.shape[0]
    vec = positions[None, :, :] - positions[:, None, :]  # x_j - x_i
    off = ~torch.eye(n, dtype=torch.bool, device=positions.device)
    safe = torch.where(off[..., None], vec, 1.0)
    t = torch.where(off[..., None, None], dipole_tensor(safe), 0.0)
    return t.permute(0, 2, 1, 3).reshape(3 * n, 3 * n)


def erfc_radial(r: Tensor, g: float) -> tuple[Tensor, Tensor, Tensor]:
    """The Ewald radial functions B, C, D of erfc(g r) / r (B' = -r C, C' = -r D):

    B = [erfc + G] / r^3, C = [3 erfc + G (3 + 2 x^2)] / r^5,
    D = [15 erfc + G (15 + 10 x^2 + 4 x^4)] / r^7,  x = g r, G = 2 x exp(-x^2) / sqrt(pi).
    """
    r2 = r * r
    x = g * r
    x2 = x * x
    gauss = (2.0 / math.sqrt(math.pi)) * x * torch.exp(-x2)
    erfc = torch.special.erfc(x)
    b = (erfc + gauss) / (r2 * r)
    c = (3.0 * erfc + gauss * (3.0 + 2.0 * x2)) / (r2 * r2 * r)
    d = (15.0 * erfc + gauss * (15.0 + x2 * (10.0 + 4.0 * x2))) / (r2 * r2 * r2 * r)
    return b, c, d


def _erfc_tensor(vec: Tensor, g: float) -> Tensor:
    """grad grad (erfc(g r) / r) = C r r^T - B 1."""
    b, c, _ = erfc_radial(torch.sqrt(vec.square().sum(-1)), g)
    eye = torch.eye(3, dtype=vec.dtype, device=vec.device)
    outer = vec[..., :, None] * vec[..., None, :]
    return c[..., None, None] * outer - b[..., None, None] * eye


class EwaldSetup:
    """q-independent parts of the Bloch-Ewald dipole matrices of one configuration.

    ``positions`` and ``cell`` (bohr) may carry autograd (forces, strain). The neighbour list
    supplies the real-space pairs; the caller chooses the splitting parameter ``g`` and
    ``k_cutoff`` so that the erfc terms beyond the list and the Gaussians beyond ``k_cutoff``
    are negligible.
    """

    def __init__(
        self,
        positions: Tensor,
        cell: Tensor,
        nl: NeighborMatrix,
        g: float,
        k_cutoff: float,
    ):
        n = positions.shape[0]
        self.n, self.g = n, g
        vec = nl.vectors(positions, cell)  # (N, M, 3), zero in padding
        safe = torch.where(nl.mask[..., None], vec, 1.0)
        self.t_real = torch.where(nl.mask[..., None, None], _erfc_tensor(safe, g), 0.0)
        self.vec = vec
        self.nl = nl

        # reciprocal lattice (rows b_i, a_i . b_j = 2 pi delta_ij) and the integer G-vectors
        # of the union of the cutoff spheres around -q for q in the first cell
        recip = 2.0 * math.pi * torch.linalg.inv(cell).T
        self.volume = torch.linalg.det(cell).abs()
        # k . a_i = 2 pi (m_i - q_i), |q_i| <= 1, so |m_i| <= k_cutoff |a_i| / (2 pi) + 1
        lengths = cell.detach().norm(dim=1)
        m_max = [math.ceil(k_cutoff * float(x) / (2.0 * math.pi)) + 1 for x in lengths]
        ranges = [torch.arange(-m, m + 1, device=positions.device) for m in m_max]
        m = torch.cartesian_prod(*ranges).to(torch.float64)  # (n_G, 3)
        reach = k_cutoff + float(recip.detach().norm(dim=1).sum())
        self.m = m[(m @ recip.detach()).norm(dim=1) <= reach]
        self.recip = recip
        self.positions = positions
        self.k_cutoff = k_cutoff

    def real_part(self, q_frac: Tensor) -> tuple[Tensor, Tensor]:
        """(real-space A (n_q, 3N, 3N), phases (N, M, n_q)) for fractional q (n_q, 3)."""
        n, n_q = self.n, q_frac.shape[0]
        dtype = torch.complex128
        # phase e^{2 pi i f . s} of the integer image shift s
        phase = torch.exp(2j * math.pi * (self.nl.shifts.to(torch.float64) @ q_frac.T))
        vals = self.t_real.reshape(n, -1, 1, 9).to(dtype) * phase[..., None]  # (N, M, n_q, 9)
        vals = vals.permute(2, 0, 1, 3)
        idx = self.nl.indices[None, :, :, None].expand(n_q, -1, -1, 9)
        real = torch.zeros(n_q, n, n + 1, 9, dtype=dtype, device=vals.device)
        real = real.scatter_add(2, idx, vals)[:, :, :n].reshape(n_q, n, n, 3, 3)
        return real.permute(0, 1, 3, 2, 4).reshape(n_q, 3 * n, 3 * n), phase

    def reciprocal_factors(self, q_frac: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """(k (n_q, n_k, 3), h (n_q, n_k), e (n_q, n_k, N), W (n_q, n_k, 3N)) with
        W_(k, jb) = h_k k_b e^{i k . x_j}, h_k = sqrt(4 pi / V exp(-k^2 / 4 g^2) / k^2), so the
        reciprocal part is -W^H W. k = (m - q) . b; h = 0 outside 0 < |k| <= k_cutoff (k = 0
        occurs only at q = 0, and dropping it there gives the regular part A_r(0) used by
        ``singular_correction``)."""
        n, n_q = self.n, q_frac.shape[0]
        k = (self.m[None] - q_frac[:, None, :]) @ self.recip  # (n_q, n_k, 3)
        k2 = k.square().sum(-1)
        keep = (k2.detach() > 1e-24) & (k2.detach() <= self.k_cutoff**2)
        k2_safe = torch.where(keep, k2, 1.0)
        c = (4.0 * math.pi / self.volume) * torch.exp(-k2_safe / (4.0 * self.g**2)) / k2_safe
        h = torch.where(keep, torch.sqrt(c), 0.0)
        e = torch.exp(1j * (k @ self.positions.T))  # e^{i k . x_j}
        w = (h[..., None] * k).to(torch.complex128)[:, :, None, :] * e[..., None]
        return k, h, e, w.reshape(n_q, -1, 3 * n)

    def self_term(self) -> float:
        return 4.0 * self.g**3 / (3.0 * math.sqrt(math.pi))

    def matrices(self, q_frac: Tensor) -> Tensor:
        """A(q) (n_q, 3N, 3N) complex for fractional q (n_q, 3)."""
        real, _ = self.real_part(q_frac)
        w = self.reciprocal_factors(q_frac)[3]
        rec = -(w.conj().transpose(1, 2) @ w)  # sum_k c k_a k_b e^{-i k x_i} e^{i k x_j}
        eye = torch.eye(3 * self.n, dtype=real.dtype, device=real.device)
        return real + rec + self.self_term() * eye

    def matrix(self, q_frac: Tensor) -> Tensor:
        """A(q) (3N, 3N) for one fractional q (3,)."""
        return self.matrices(q_frac[None])[0]

    def bytes_per_q(self) -> int:
        """Rough peak memory of one q in ``matrices`` (the reciprocal factor dominates)."""
        return 16 * (self.m.shape[0] * 3 * self.n * 2 + 9 * self.t_real.shape[1] * self.n * 2)


def _block_sum(x: Tensor) -> Tensor:
    """(3N, 3N) -> (3, 3): sum over the atom indices of both block axes."""
    n = x.shape[0] // 3
    return x.reshape(n, 3, n, 3).sum((0, 2))


def singular_correction(
    setup: EwaldSetup,
    fractions: Tensor,
    weights: Tensor,
    a: Tensor,
    mu: Tensor,
    sigma_scale: float = 1.0,
) -> Tensor:
    """Correction to the q-grid average of the trace for the non-analytic q -> 0 limit.

    As q -> 0 the G = 0 reciprocal term tends to B = beta u u^T, beta = -4 pi / V,
    u = qhat (x) 1_N, which depends on the direction of q. With M = D A_r(0) (A_r the matrix
    without that term) the trace therefore jumps by

        F(qhat) = Tr[(M + beta D u u^T)^3] - Tr[M^3]
                = beta^3 alpha^3 + qhat^T K qhat,
        K = 3 beta S2 + 3 beta^2 alpha S1,   S1 = bsum(D A D), S2 = bsum(D A D A D),

    alpha = sum_i a_i, bsum the 3 x 3 sum over atom blocks. On a grid this discontinuity gives
    an O(1/N_q) error. The auxiliary function F(unit(q + G)) exp(-|q + G|^2 / sigma^2), summed
    over G, has the same singularity; its grid average is subtracted and its exact integral
    (sphere average of F times pi^(3/2) sigma^3 / V_BZ) added. Returns the correction to
    sum_r mu_r <Tr[(D_r A)^3]>_q, float64.
    """
    f64 = torch.float64
    a0 = setup.matrix(torch.zeros(3, dtype=f64, device=a.device)).real
    beta = -4.0 * math.pi / setup.volume
    recip = setup.recip
    b_min = float(recip.detach().norm(dim=1).min())
    sigma = sigma_scale * b_min
    m_max = math.ceil(6.0 * sigma / b_min) + 1
    rng = torch.arange(-m_max, m_max + 1, device=a.device, dtype=f64)
    g = torch.cartesian_prod(rng, rng, rng) @ recip  # (n_G, 3)
    p = (fractions @ recip)[:, None, :] + g[None]  # (n_q, n_G, 3)
    p2 = p.square().sum(-1)
    gauss = torch.exp(-p2 / sigma**2)
    unit = p / torch.sqrt(p2)[..., None]
    v_bz = (2.0 * math.pi) ** 3 / setup.volume
    integral_scale = math.pi**1.5 * sigma**3 / v_bz

    total = torch.zeros((), dtype=f64, device=a.device)
    for r in range(mu.shape[0]):
        d = a[:, r].to(f64).repeat_interleave(3)
        dad = d[:, None] * a0 * d[None, :]
        s1 = _block_sum(dad)
        s2 = _block_sum(dad @ (a0 * d[None, :]))
        alpha = a[:, r].to(f64).sum()
        k = 3.0 * beta * s2 + 3.0 * beta**2 * alpha * s1
        k = 0.5 * (k + k.T)
        c0 = beta**3 * alpha**3
        f = c0 + torch.einsum("qga,ab,qgb->qg", unit, k, unit)
        grid = (weights[:, None] * f * gauss).sum()
        integral = (c0 + torch.trace(k) / 3.0) * integral_scale
        total = total + mu[r] * (integral - grid)
    return total


def trace_energy(matrices: Tensor, weights: Tensor, a: Tensor, mu: Tensor) -> Tensor:
    """-(1/18) sum_r mu_r sum_q w_q Re Tr[(D_r A_q)^3] (without s9), float64 scalar.

    ``matrices``: A_q (n_q, 3N, 3N); ``a`` (N, R) and ``mu`` (R,) the separable C9 factors.
    """
    d = a.to(torch.float64).repeat_interleave(3, dim=0).T  # (R, 3N)
    m = d[None, :, :, None].to(matrices.dtype) * matrices[:, None]  # (n_q, R, 3N, 3N): D_r A
    tr = ((m @ m) * m.transpose(-1, -2)).sum((-1, -2)).real  # (n_q, R)
    return -(weights[:, None] * mu.to(torch.float64) * tr).sum() / 18.0
