"""EEQ-2019 atomic charges (multicharge 0.5.0, as used by dftd4 4.3.0).

The charges solve the constrained electronegativity equilibration

    [A 1; 1^T 0] [q; lambda] = [x; Q],   x_i = -chi_i + kcnchi_i CN_i / sqrt(CN_i + 1e-14),

with the EEQ coordination number (erf counting, kcn = 7.5, 25 bohr, capped smoothly at 8). The
Coulomb matrix uses Gaussian charges of width rad_i:

- molecules: A_ij = erf(r_ij / sqrt(rad_i^2 + rad_j^2)) / r_ij, A_ii = eta_i + sqrt(2/pi) / rad_i;
- 3D cells: multicharge's Ewald form. For every pair the minimum images (Wigner-Seitz, ties
  within 0.01 bohr^2 averaged) carry a direct sum of erf(g r)/r - erf(alpha r)/r over a fixed
  5 x 5 x 5 block of translations, plus the reciprocal sum over the 124 shortest G vectors of
  the 5 x 5 x 5 reciprocal block, and A_ii gets -2 alpha / sqrt(pi). The splitting alpha is
  multicharge's bisection on the lattice (held fixed under strain). These fixed-range sums are
  the oracle's definition, so they are replicated rather than replaced by a converged Ewald
  sum. (multicharge looks up the Wigner-Seitz image of a self pair one entry before the true
  one in its translation table; that is replicated too.)

The reciprocal part is exactly periodic, so it factorises into structure factors (an N x 124
matmul). The direct part is a reduction over the 125 translations per image pair, evaluated in
pair chunks with a hand-written backward (``_DirectCoulomb``), so its memory stays bounded by
one chunk and nothing is recomputed under autograd. The solve is differentiated by autograd
(``torch.linalg.solve``).
"""

import math
from dataclasses import dataclass
from functools import cache, partial

import torch
from torch import Tensor

from ..core.coordination import coordination_number
from .model import erf_count

CN_CUTOFF = 25.0
KCN = 7.5
CN_MAX = 8.0
_SQRT2PI = math.sqrt(2.0 / math.pi)
_TWO_OVER_SQRTPI = 2.0 / math.sqrt(math.pi)
_WS_THRESHOLD = math.sqrt(2.220446049250313e-16)
_WS_TOLERANCE = 0.01
_PAIR_CHUNK = 16384


def eeq_coordination_number(
    vectors: Tensor, mask: Tensor, rcov_centre: Tensor, rcov_neighbor: Tensor
) -> Tensor:
    """sum_m 0.5 (1 + erf(-7.5 (r_im - Rc) / Rc)) within 25 bohr, then the smooth cap
    log(1 + e^8) - log(1 + e^(8 - CN))."""
    count = partial(erf_count, kcn=KCN)
    cn = coordination_number(vectors, mask, rcov_centre, rcov_neighbor, CN_CUTOFF, count)
    return math.log1p(math.exp(CN_MAX)) - torch.nn.functional.softplus(CN_MAX - cn)


def electronegativity_vector(cn: Tensor, chi: Tensor, kcnchi: Tensor) -> Tensor:
    return -chi + kcnchi / torch.sqrt(cn + 1e-14) * cn


# --------------------------------------------------------------------------------------------
# lattice helpers (host side, float64; they depend on the cell only)


def _lattice_points(rep: tuple[int, int, int], origin: bool) -> list[tuple[int, int, int]]:
    """Integer translations in mctc-lib's ``get_lattice_points`` order."""
    out = []
    for ix in range(rep[0] + 1):
        for iy in range(rep[1] + 1):
            for iz in range(rep[2] + 1):
                if not origin and ix == iy == iz == 0:
                    continue
                for jx in (1, -1) if ix > 0 else (1,):
                    for jy in (1, -1) if iy > 0 else (1,):
                        for jz in (1, -1) if iz > 0 else (1,):
                            out.append((ix * jx, iy * jy, iz * jz))
    return out


def _rec_term(g: float, alpha: float, volume: float) -> float:
    return 4.0 * math.pi * math.exp(-0.25 * g * g / (alpha * alpha)) / (volume * g * g)


def _dir_term(r: float, alpha: float) -> float:
    return math.erfc(alpha * r) / r


def ewald_alpha(cell: Tensor) -> float:
    """multicharge ``get_alpha``: bisection for the splitting at which the truncation tails of
    the direct (2-3 shortest lattice lengths) and reciprocal (4-5 shortest reciprocal lengths)
    sums agree to sqrt(eps). ``cell`` rows are lattice vectors (bohr)."""
    lat = cell.detach().to("cpu", torch.float64)
    volume = abs(float(torch.linalg.det(lat)))
    rec = 2.0 * math.pi * torch.linalg.inv(lat).T
    rlen = math.sqrt(float(rec.square().sum(-1).min()))
    dlen = math.sqrt(float(lat.square().sum(-1).min()))
    tol = _WS_THRESHOLD

    def diff(a: float) -> float:
        return (_rec_term(4 * rlen, a, volume) - _rec_term(5 * rlen, a, volume)) - (
            _dir_term(2 * dlen, a) - _dir_term(3 * dlen, a)
        )

    alpha0 = 1e-8
    alpha = alpha0
    d = diff(alpha)
    while d < -tol and alpha <= 1.7e308:
        alpha *= 2.0
        d = diff(alpha)
    if alpha > 1.7e308 or alpha == alpha0:
        return 0.25
    lo = 0.5 * alpha
    while d < tol and alpha <= 1.7e308:
        alpha *= 2.0
        d = diff(alpha)
    if alpha > 1.7e308:
        return 0.25
    hi = alpha
    alpha = 0.5 * (lo + hi)
    d = diff(alpha)
    it = 0
    while abs(d) > tol and it <= 30:
        if d < 0:
            lo = alpha
        else:
            hi = alpha
        alpha = 0.5 * (lo + hi)
        d = diff(alpha)
        it += 1
    return 0.25 if it > 30 else alpha


@dataclass(frozen=True)
class ImagePairs:
    """Wigner-Seitz images of the pairs i <= j: entry e is atom pair (i[e], j[e]) with the
    lattice shift ``shift[e]`` (vector x_j - x_i + shift @ cell) and weight 1 / n_images."""

    i: Tensor  # (E,) int64
    j: Tensor  # (E,) int64
    shift: Tensor  # (E, 3) integer shifts, as float
    weight: Tensor  # (E,)


def central_cell_offsets(positions: Tensor, cell: Tensor) -> Tensor:
    """Integer lattice offsets k with x - k @ cell in the central cell, by dftd4's
    ``wrap_to_central_cell`` rule (fractional [0, 1) up to a 1e-14 tolerance), (N, 3)."""
    frac = positions @ torch.linalg.inv(cell)
    eps = 1e-14
    k = torch.zeros_like(frac)
    k = torch.where(frac < -eps, -torch.ceil(-frac), k)
    k = torch.where(frac > 1.0 + eps, torch.floor(frac), k)
    k = torch.where((frac - 1.0).abs() < eps, 1.0, k)
    return k


def wigner_seitz_images(positions: Tensor, cell: Tensor) -> ImagePairs:
    """multicharge ``new_wignerseitz_cell`` over the 27 translations of the +-1 block: per pair
    the minimum images of x_i - x_j - T (r^2 >= sqrt(eps)) within 0.01 bohr^2 of the minimum.

    dftd4 wraps all positions into the central cell first, and the +-1 block only finds the
    minimum images of wrapped positions, so the search runs on wrapped positions and the wrap
    offsets are folded into the returned shifts (valid for the caller's positions)."""
    with torch.no_grad():
        lat = cell.detach().to(torch.float64)
        offsets = central_cell_offsets(positions.detach().to(torch.float64), lat)
        pos = positions.detach().to(torch.float64) - offsets @ lat
        n, device = pos.shape[0], pos.device
        trans_int = torch.tensor(_lattice_points((1, 1, 1), True), device=device)
        trans = trans_int.to(torch.float64) @ lat  # (27, 3), entry 0 is the origin
        iu, ju = torch.triu_indices(n, n, device=device)
        parts = []
        for start in range(0, iu.numel(), _PAIR_CHUNK):
            i, j = iu[start : start + _PAIR_CHUNK], ju[start : start + _PAIR_CHUNK]
            d = pos[i] - pos[j]
            r2 = (d[:, None, :] - trans).square().sum(-1)  # (P, 27)
            keep = r2 >= _WS_THRESHOLD
            r2 = torch.where(keep, r2, torch.inf)
            chosen = keep & (r2 <= r2.amin(-1, keepdim=True) + _WS_TOLERANCE)
            # self pairs skip the origin, which shifts multicharge's lookup by one entry
            col = torch.arange(trans.shape[0], device=device)
            col = torch.where((i == j)[:, None], col - 1, col).expand_as(chosen)
            p, k = chosen.nonzero(as_tuple=True)
            weight = 1.0 / chosen.sum(-1).to(torch.float64)
            parts.append((i[p], j[p], trans_int[col[p, k]], weight[p]))
        i, j, shift, weight = (torch.cat(x) for x in zip(*parts, strict=True))
        # wrapped x'_j - x'_i + T = x_j - x_i + (T - k_j + k_i) @ cell
        shift = shift.to(torch.float64) - offsets[j] + offsets[i]
    # |x_j - x_i + T| = |x_i - x_j - T|: the image vector has the candidate's length
    return ImagePairs(i=i, j=j, shift=shift.to(positions.dtype), weight=weight.to(positions.dtype))


def _direct_sum(vec: Tensor, block: Tensor, gam: Tensor, alpha: Tensor) -> Tensor:
    """sum_T [erf(g r) - erf(alpha r)] / r over the translations ``block`` (T, 3), r = |vec + T|,
    skipping r < sqrt(eps); vec (P, 3), gam (P,)."""
    d = vec[:, None, :] + block
    r2 = d.square().sum(-1)
    keep = r2 >= _WS_THRESHOLD * _WS_THRESHOLD
    r = torch.sqrt(torch.where(keep, r2, 1.0))
    term = (torch.erf(gam[:, None] * r) - torch.erf(alpha * r)) / r
    return torch.where(keep, term, 0.0).sum(-1)


def _direct_sum_grad(
    vec: Tensor, block: Tensor, gam: Tensor, alpha: Tensor, g: Tensor
) -> tuple[Tensor, Tensor]:
    """Derivatives of sum_e g_e ``_direct_sum``_e with respect to vec (P, 3) and block (T, 3):
    d f / d r = 2 / sqrt(pi) (gam exp(-gam^2 r^2) - alpha exp(-alpha^2 r^2)) / r - f / r."""
    d = vec[:, None, :] + block
    r2 = d.square().sum(-1)
    keep = r2 >= _WS_THRESHOLD * _WS_THRESHOLD
    r2 = torch.where(keep, r2, 1.0)
    r = torch.sqrt(r2)
    gr = gam[:, None]
    f = (torch.erf(gr * r) - torch.erf(alpha * r)) / r
    gauss = gr * torch.exp(-gr * gr * r2) - alpha * torch.exp(-alpha * alpha * r2)
    dfdr = (_TWO_OVER_SQRTPI * gauss - f) / r
    dd = torch.where(keep, g[:, None] * dfdr / r, 0.0)[..., None] * d  # (P, T, 3)
    return dd.sum(1), dd.sum(0)


@cache
def _compiled(fn):
    return torch.compile(fn, fullgraph=True, dynamic=True)


def _maybe_compiled(fn, compile: bool):
    return _compiled(fn) if compile else fn


@cache
def _direct_block_int() -> tuple[tuple[int, int, int], ...]:
    return tuple(_lattice_points((2, 2, 2), True))


def _direct_block(device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor(_direct_block_int(), device=device, dtype=dtype)


def _scatter_symmetric(n: int, pairs: ImagePairs, values: Tensor) -> Tensor:
    """(n, n) matrix with values at (i, j) and (j, i) (once on the diagonal)."""
    a = values.new_zeros(n * n)
    a = a.index_add(0, pairs.i * n + pairs.j, values)
    off = pairs.i != pairs.j
    a = a.index_add(0, pairs.j[off] * n + pairs.i[off], values[off])
    return a.reshape(n, n)


class _DirectCoulomb(torch.autograd.Function):
    """Direct-space part of the periodic Coulomb matrix, w_e sum_T [erf(g r) - erf(alpha r)] / r
    per image pair e over the 5 x 5 x 5 block, with an analytic backward. Both directions are
    reductions over the block evaluated in pair chunks (``compile``: fused, dynamic shapes), so
    no (pairs x 125) tensor is kept."""

    @staticmethod
    def forward(ctx, pos, cell, pairs, gam_table, alpha, compile):
        n = pos.shape[0]
        fn = _maybe_compiled(_direct_sum, compile)
        block = _direct_block(cell.device, cell.dtype) @ cell
        vec = pos[pairs.j] - pos[pairs.i] + pairs.shift @ cell
        gam = gam_table[pairs.i, pairs.j]
        values = torch.cat(
            [
                fn(vec[s : s + _PAIR_CHUNK], block, gam[s : s + _PAIR_CHUNK], alpha)
                for s in range(0, vec.shape[0], _PAIR_CHUNK)
            ]
        )
        ctx.save_for_backward(vec, block, gam, alpha)
        ctx.pairs, ctx.n, ctx.compile = pairs, n, compile
        return _scatter_symmetric(n, pairs, pairs.weight * values)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_a,) = grad_outputs
        vec, block, gam, alpha = ctx.saved_tensors
        pairs = ctx.pairs
        # d(sum grad_a * A) / d(entry e) = grad_a[i, j] + grad_a[j, i] off the diagonal
        g = grad_a[pairs.i, pairs.j]
        off = pairs.i != pairs.j
        g = pairs.weight * torch.where(off, g + grad_a[pairs.j, pairs.i], g)
        fn = _maybe_compiled(_direct_sum_grad, ctx.compile)
        grad_vec, grad_block = [], torch.zeros_like(block)
        for s in range(0, vec.shape[0], _PAIR_CHUNK):
            sl = slice(s, s + _PAIR_CHUNK)
            dv, db = fn(vec[sl], block, gam[sl], alpha, g[sl])
            grad_vec.append(dv)
            grad_block = grad_block + db
        dvec = torch.cat(grad_vec)
        # vec = x_j - x_i + shift @ cell, block = ints @ cell
        grad_pos = torch.zeros((ctx.n, 3), dtype=dvec.dtype, device=dvec.device)
        grad_pos = grad_pos.index_add(0, pairs.j, dvec).index_add(0, pairs.i, -dvec)
        ints = _direct_block(block.device, block.dtype)
        grad_cell = pairs.shift.T @ dvec + ints.T @ grad_block
        return grad_pos, grad_cell, None, None, None, None


def _reciprocal_vectors(cell: Tensor) -> Tensor:
    """The 124 G vectors of the 5 x 5 x 5 reciprocal block (origin excluded), (124, 3)."""
    ints = torch.tensor(_lattice_points((2, 2, 2), False), device=cell.device, dtype=cell.dtype)
    return ints @ (2.0 * torch.pi * torch.linalg.inv(cell).mT)


def coulomb_matrix_reciprocal(
    pos: Tensor, cell: Tensor, rad: Tensor, eta: Tensor, alpha: Tensor
) -> Tensor:
    """(n, n) reciprocal-space part plus the diagonal of the periodic EEQ Coulomb matrix (the
    direct part is ``_DirectCoulomb``); ``alpha`` a 0-d tensor. Differentiable in ``pos`` and
    ``cell``, and free of host syncs (it is part of the compiled graph stage)."""
    g = _reciprocal_vectors(cell)
    g2 = g.square().sum(-1)
    volume = torch.linalg.det(cell).abs()
    f = (4.0 * torch.pi) * torch.exp(-0.25 * g2 / (alpha * alpha)) / (g2 * volume)
    phase = pos @ g.T  # (n, 124)
    cos, sin = torch.cos(phase), torch.sin(phase)
    a_rec = (cos * f) @ cos.T + (sin * f) @ sin.T
    diag = eta + _SQRT2PI / rad - 2.0 * alpha / math.sqrt(math.pi)
    return a_rec + torch.diag(diag)


def direct_coulomb(
    pos: Tensor, cell: Tensor, rad: Tensor, alpha: Tensor, pairs: ImagePairs, compile: bool
) -> Tensor:
    """(n, n) direct-space part of the periodic EEQ Coulomb matrix (``_DirectCoulomb``)."""
    gam_table = 1.0 / torch.sqrt(rad[:, None].square() + rad[None, :].square())
    return _DirectCoulomb.apply(pos, cell, pairs, gam_table, alpha, compile)


def coulomb_matrix_periodic(
    pos: Tensor,
    cell: Tensor,
    rad: Tensor,
    eta: Tensor,
    alpha: float,
    pairs: ImagePairs,
    compile: bool = False,
) -> Tensor:
    """(n, n) EEQ Coulomb matrix of a 3D cell; differentiable in ``pos`` and ``cell``."""
    alpha_t = torch.tensor(alpha, dtype=pos.dtype, device=pos.device)
    a_dir = direct_coulomb(pos, cell, rad, alpha_t, pairs, compile)
    return a_dir + coulomb_matrix_reciprocal(pos, cell, rad, eta, alpha_t)


def coulomb_matrix_molecular(pos: Tensor, rad: Tensor, eta: Tensor) -> Tensor:
    n = pos.shape[0]
    vec = pos[None, :, :] - pos[:, None, :]
    off = ~torch.eye(n, dtype=torch.bool, device=pos.device)
    r = torch.sqrt(torch.where(off, vec.square().sum(-1), 1.0))
    gam = 1.0 / torch.sqrt(rad[:, None].square() + rad[None, :].square())
    a = torch.where(off, torch.erf(gam * r) / r, 0.0)
    return a + torch.diag(eta + _SQRT2PI / rad)


def solve_charges(a: Tensor, x: Tensor, total_charge: float | Tensor) -> Tensor:
    """q from the bordered system [A 1; 1^T 0] [q; lambda] = [x; Q] (Q a float or 0-d tensor)."""
    n = a.shape[0]
    ones = a.new_ones(n, 1)
    top = torch.cat([a, ones], 1)
    bottom = torch.cat([ones.T, a.new_zeros(1, 1)], 1)
    rhs = torch.cat([x, torch.as_tensor(total_charge, dtype=x.dtype, device=x.device).reshape(1)])
    return torch.linalg.solve(torch.cat([top, bottom], 0), rhs)[:n]
