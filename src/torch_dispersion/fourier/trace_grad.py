"""Analytic gradients of the q-grid trace energy (the ``trace_gradient="analytic"`` backend).

    E = -(1/18) sum_q w_q sum_r mu_r Re Tr[M_r^3],   M_r = D_r A(q),  D_r = diag(a_r) (x) 1_3

Without autograd, per q-chunk:

* dE/dA = Gamma = -(w_q / 6) sum_r mu_r (M_r^2 D_r)^T, i.e. dE = Re sum Gamma_uv dA_uv;
* dE/dd_u = -(w_q / 6) mu_r Re (A M_r^2)_uu, summed over the 3 components of each atom;
* real space, A_(ia, j_m b) += T_erfc(v_im) e^{2 pi i f.s_im}: with H_im = Re sum_q
  Gamma[(i,:), (j_m,:)] phase, dE/dv_im = -D (v.Hv) v + C (H + H^T) v + C tr(H) v, from
  T_erfc = C v v^T - B 1, B' = -r C, C' = -r D (``dipole.erfc_radial``);
* reciprocal space, A_rec = -W^H W with W_(k, jb) = h_k k_b e^{i k.x_j}: dE = -Re sum dW Z,
  Z = conj(W) (Gamma + Gamma^H). Positions enter only through the phases, dE/dx_j =
  sum_k k Im(U_kj) with U_kj = sum_b W Z. Under a strain, k.x is invariant and only h_k and
  k_b change (dk_b = -eps_ba k_a, d ln V = tr eps), which gives

      dE/deps_xy = sum_k Re[ zeta_k (delta_xy / 2 - alpha_k k_x k_y) + Y_kx k_y ],
      zeta_k = sum_j U_kj, Y_kb = h_k sum_j Z_(k, jb) e^{i k.x_j}, alpha_k = 1/(4 g^2) + 1/k^2.

The q -> 0 correction (``dipole.singular_correction``, one extra matrix) stays on autograd.
"""

from typing import NamedTuple

import torch
from torch import Tensor

from .dipole import EwaldSetup, erfc_radial


class TraceGradients(NamedTuple):
    energy: Tensor  # () Hartree, without s9
    grad_pos: Tensor  # (N, 3) explicit dE/dx
    virial: Tensor  # (3, 3) explicit dE/d(strain)
    grad_a: Tensor  # (N, R)


def trace_energy_and_grad(
    setup: EwaldSetup,
    fractions: Tensor,
    weights: Tensor,
    a: Tensor,
    mu: Tensor,
    chunk: int,
) -> TraceGradients:
    """Energy and gradients of the grid trace. ``setup`` must be built from detached
    positions and cell; ``a`` (N, R), ``mu`` (R,) detached, float64."""
    f64, c128 = torch.float64, torch.complex128
    n, rank = setup.n, mu.shape[0]
    device = a.device
    d = a.to(f64).repeat_interleave(3, dim=0).T  # (R, 3N)
    mu = mu.to(f64)
    energy = torch.zeros((), dtype=f64, device=device)
    grad_pos = torch.zeros((n, 3), dtype=f64, device=device)
    virial = torch.zeros((3, 3), dtype=f64, device=device)
    grad_d = torch.zeros((rank, 3 * n), dtype=f64, device=device)
    h_real = torch.zeros((n, setup.nl.width, 3, 3), dtype=f64, device=device)
    eye3 = torch.eye(3, dtype=f64, device=device)
    alpha_g = 1.0 / (4.0 * setup.g**2)

    for start in range(0, fractions.shape[0], chunk):
        f, w_q = fractions[start : start + chunk], weights[start : start + chunk]
        c = f.shape[0]
        real, phase = setup.real_part(f)
        k, h, e, w = setup.reciprocal_factors(f)
        mat = real - (w.conj().transpose(1, 2) @ w)
        mat = mat + setup.self_term() * torch.eye(3 * n, dtype=c128, device=device)

        gamma = torch.zeros_like(mat)
        for r in range(rank):
            m = d[r][None, :, None].to(c128) * mat  # D_r A
            m2 = m @ m
            tr = (m2 * m.transpose(1, 2)).sum((1, 2)).real
            energy = energy - (w_q * mu[r] * tr).sum() / 18.0
            coef = (-(w_q * mu[r]) / 6.0)[:, None, None]
            # dTr(M^3)/dA = 3 (M^2 D)^T, dTr(M^3)/dd_u = 3 (A M^2)_uu
            gamma = gamma + coef * (m2 * d[r][None, None, :].to(c128)).transpose(1, 2)
            diag = (mat * m2.transpose(1, 2)).sum(-1).real  # (c, 3N)
            grad_d[r] = grad_d[r] + (coef[:, :, 0] * diag).sum(0)

        # real space: gather Gamma's (i, j_m) blocks, contract with the Bloch phases
        g4 = gamma.reshape(c, n, 3, n, 3).permute(0, 1, 3, 2, 4)  # (c, N, N, 3, 3)
        g4 = torch.nn.functional.pad(g4, (0, 0, 0, 0, 0, 1))  # padding index N -> zeros
        rows = torch.arange(n, device=device)[:, None]
        blocks = g4[:, rows, setup.nl.indices]  # (c, N, M, 3, 3)
        h_real = h_real + (blocks * phase.permute(2, 0, 1)[..., None, None]).sum(0).real

        # reciprocal space
        z = w.conj() @ (gamma + gamma.conj().transpose(1, 2))  # (c, n_k, 3N)
        wz = (w * z).reshape(c, -1, n, 3)
        u = wz.sum(-1)  # (c, n_k, N)
        grad_pos = grad_pos + torch.einsum("qkj,qkx->jx", u.imag, k)
        zeta = u.sum(-1)  # (c, n_k)
        y = h[..., None] * torch.einsum("qkjb,qkj->qkb", z.reshape(c, -1, n, 3), e)
        k2 = k.square().sum(-1)
        alpha = alpha_g + 1.0 / torch.where(h > 0, k2, 1.0)
        kk = k[..., :, None] * k[..., None, :]
        virial = virial + (
            0.5 * zeta.real.sum() * eye3
            - torch.einsum("qk,qkxy->xy", (alpha * zeta).real, kk)
            + torch.einsum("qkx,qky->xy", y.real, k)
        )

    # real-space pair derivatives
    vec, mask = setup.vec, setup.nl.mask
    r = torch.sqrt(torch.where(mask, vec.square().sum(-1), 1.0))
    _, cc, dd = erfc_radial(r, setup.g)
    hv = torch.einsum("nmab,nmb->nma", h_real + h_real.transpose(-1, -2), vec)
    vhv = torch.einsum("nma,nmab,nmb->nm", vec, h_real, vec)
    tr_h = torch.diagonal(h_real, dim1=-2, dim2=-1).sum(-1)
    g = -dd[..., None] * vhv[..., None] * vec + cc[..., None] * (hv + tr_h[..., None] * vec)
    g = torch.where(mask[..., None], g, 0.0)
    grad_pos = grad_pos - g.sum(1)
    grad_pos = grad_pos.index_add(0, setup.nl.indices.clamp(max=n - 1).flatten(), g.flatten(0, 1))
    virial = virial + torch.einsum("nmx,nmy->xy", vec, g)
    grad_a = grad_d.reshape(rank, n, 3).sum(-1).T
    return TraceGradients(energy, grad_pos, virial, grad_a)
