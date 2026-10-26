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

Everything works on a batch of systems (``core.batch.Batch``): the image pairs and the direct sum
over a flat list of pairs of all systems (each with its own cell and splitting), the Coulomb
matrices and the bordered solves in a padded layout (S, n_max, n_max), where empty rows are
decoupled with a unit diagonal and get q = 0.

The reciprocal part is exactly periodic, so it factorises into structure factors (an n x 124
matmul per system). The direct part is a reduction over the 125 translations per image pair,
evaluated in pair chunks with a hand-written backward (``_DirectCoulomb``), so its memory stays
bounded by one chunk and nothing is recomputed under autograd. The solve is differentiated by
autograd (``torch.linalg.solve``).
"""

import math
from dataclasses import dataclass
from functools import cache, partial

import torch
from torch import Tensor

from ..core.batch import Batch
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
    """Wigner-Seitz images of the pairs i <= j of every system of a batch: entry e is the atom
    pair (i[e], j[e]) (flat atom indices) of system ``system[e]`` with the lattice shift
    ``shift[e]`` (vector x_j - x_i + shift @ cell) and weight 1 / n_images."""

    i: Tensor  # (E,) int64
    j: Tensor  # (E,) int64
    system: Tensor  # (E,) int64
    shift: Tensor  # (E, 3) integer shifts, as float
    weight: Tensor  # (E,)


def ewald_alphas(cells: Tensor) -> Tensor:
    """``ewald_alpha`` of each cell (S, 3, 3) as an (S,) float64 tensor (one host transfer)."""
    host = cells.detach().to("cpu", torch.float64)
    alphas = [ewald_alpha(c) for c in host]
    return torch.tensor(alphas, dtype=torch.float64, device=cells.device)


def central_cell_offsets(positions: Tensor, cells: Tensor) -> Tensor:
    """Integer lattice offsets k with x - k @ cell in the central cell, by dftd4's
    ``wrap_to_central_cell`` rule (fractional [0, 1) up to a 1e-14 tolerance), (N, 3).
    ``cells`` is one cell (3, 3) or the cell of each atom (N, 3, 3)."""
    inv = torch.linalg.inv(cells)
    frac = positions @ inv if inv.dim() == 2 else (positions[:, None, :] @ inv)[:, 0]
    eps = 1e-14
    k = torch.zeros_like(frac)
    k = torch.where(frac < -eps, -torch.ceil(-frac), k)
    k = torch.where(frac > 1.0 + eps, torch.floor(frac), k)
    k = torch.where((frac - 1.0).abs() < eps, 1.0, k)
    return k


def _upper_pairs(counts: tuple[int, ...], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """(i, j, system) of all pairs i <= j within each system, flat atom indices."""
    n = torch.tensor(counts, device=device)
    n_pairs = n * (n + 1) // 2
    system = torch.repeat_interleave(torch.arange(len(counts), device=device), n_pairs)
    k = torch.arange(system.numel(), device=device) - (torch.cumsum(n_pairs, 0) - n_pairs)[system]
    # k = j (j + 1) / 2 + i with 0 <= i <= j: invert the triangular numbers, then fix rounding
    j = ((torch.sqrt(8.0 * k.to(torch.float64) + 1.0) - 1.0) / 2.0).floor().to(torch.int64)
    j = torch.where((j + 1) * (j + 2) // 2 <= k, j + 1, j)
    j = torch.where(j * (j + 1) // 2 > k, j - 1, j)
    i = k - j * (j + 1) // 2
    start = (torch.cumsum(n, 0) - n)[system]
    return start + i, start + j, system


def wigner_seitz_images(positions: Tensor, cells: Tensor, batch: Batch) -> ImagePairs:
    """multicharge ``new_wignerseitz_cell`` over the 27 translations of the +-1 block, for every
    system of the batch: per pair the minimum images of x_i - x_j - T (r^2 >= sqrt(eps)) within
    0.01 bohr^2 of the minimum. ``positions`` (N, 3), ``cells`` (S, 3, 3), bohr.

    dftd4 wraps all positions into the central cell first, and the +-1 block only finds the
    minimum images of wrapped positions, so the search runs on wrapped positions and the wrap
    offsets are folded into the returned shifts (valid for the caller's positions)."""
    with torch.no_grad():
        lat = cells.detach().to(torch.float64)
        device = lat.device
        offsets = central_cell_offsets(positions.detach().to(torch.float64), lat[batch.system])
        pos = positions.detach().to(torch.float64) - (offsets[:, None, :] @ lat[batch.system])[:, 0]
        trans_int = torch.tensor(_lattice_points((1, 1, 1), True), device=device)
        trans_f = trans_int.to(torch.float64)
        iu, ju, su = _upper_pairs(batch.counts, device)
        parts = []
        for start in range(0, iu.numel(), _PAIR_CHUNK):
            i, j = iu[start : start + _PAIR_CHUNK], ju[start : start + _PAIR_CHUNK]
            s = su[start : start + _PAIR_CHUNK]
            trans = trans_f @ lat[s]  # (P, 27, 3), entry 0 is the origin
            d = pos[i] - pos[j]
            r2 = (d[:, None, :] - trans).square().sum(-1)  # (P, 27)
            keep = r2 >= _WS_THRESHOLD
            r2 = torch.where(keep, r2, torch.inf)
            chosen = keep & (r2 <= r2.amin(-1, keepdim=True) + _WS_TOLERANCE)
            # self pairs skip the origin, which shifts multicharge's lookup by one entry
            col = torch.arange(trans.shape[1], device=device)
            col = torch.where((i == j)[:, None], col - 1, col).expand_as(chosen)
            p, k = chosen.nonzero(as_tuple=True)
            weight = 1.0 / chosen.sum(-1).to(torch.float64)
            parts.append((i[p], j[p], s[p], trans_int[col[p, k]], weight[p]))
        i, j, s, shift, weight = (torch.cat(x) for x in zip(*parts, strict=True))
        # wrapped x'_j - x'_i + T = x_j - x_i + (T - k_j + k_i) @ cell
        shift = shift.to(torch.float64) - offsets[j] + offsets[i]
    # |x_j - x_i + T| = |x_i - x_j - T|: the image vector has the candidate's length
    return ImagePairs(
        i=i, j=j, system=s, shift=shift.to(positions.dtype), weight=weight.to(positions.dtype)
    )


def _translations(ints: Tensor, cell: Tensor) -> Tensor:
    """ints (T, 3) @ cell (P, 3, 3) -> (P, T, 3), written as a broadcast sum over three terms
    so that the compiled reductions fuse it instead of materialising a batched matmul."""
    return (ints[None, :, :, None] * cell[:, None, :, :]).sum(-2)


def _direct_sum(vec: Tensor, cell: Tensor, ints: Tensor, gam: Tensor, alpha: Tensor) -> Tensor:
    """sum_T [erf(g r) - erf(alpha r)] / r over the translations T = ints @ cell, r = |vec + T|,
    skipping r < sqrt(eps). Per pair: vec (P, 3), cell (P, 3, 3), gam (P,), alpha (P,);
    ints (T, 3)."""
    d = vec[:, None, :] + _translations(ints, cell)
    r2 = d.square().sum(-1)
    keep = r2 >= _WS_THRESHOLD * _WS_THRESHOLD
    r = torch.sqrt(torch.where(keep, r2, 1.0))
    term = (torch.erf(gam[:, None] * r) - torch.erf(alpha[:, None] * r)) / r
    return torch.where(keep, term, 0.0).sum(-1)


def _direct_sum_coef(
    vec: Tensor, cell: Tensor, ints: Tensor, gam: Tensor, alpha: Tensor, g: Tensor
) -> Tensor:
    """(P, T) coefficients c with d(g_e ``_direct_sum``_e) / d(vec + T) = c_eT (vec + T)_e:
    c = g f'(r) / r, f'(r) = 2 / sqrt(pi) (gam exp(-gam^2 r^2) - alpha exp(-alpha^2 r^2)) / r
    - f / r. The only place the backward evaluates the special functions."""
    d = vec[:, None, :] + _translations(ints, cell)
    r2 = d.square().sum(-1)
    keep = r2 >= _WS_THRESHOLD * _WS_THRESHOLD
    r2 = torch.where(keep, r2, 1.0)
    r = torch.sqrt(r2)
    gr, ar = gam[:, None], alpha[:, None]
    f = (torch.erf(gr * r) - torch.erf(ar * r)) / r
    gauss = gr * torch.exp(-gr * gr * r2) - ar * torch.exp(-ar * ar * r2)
    dfdr = (_TWO_OVER_SQRTPI * gauss - f) / r
    return torch.where(keep, g[:, None] * dfdr / r, 0.0)


def _direct_sum_grad(
    vec: Tensor, cell: Tensor, ints: Tensor, gam: Tensor, alpha: Tensor, g: Tensor
) -> tuple[Tensor, Tensor]:
    """Per pair, the derivatives of g_e ``_direct_sum``_e with respect to vec (P, 3) and, through
    the translations T = ints @ cell, to the cell (P, 3, 3), from the coefficients by matmuls:

        d/d vec = sum_T c_T (vec + T) = (sum_T c_T) vec + (c @ ints) @ cell,
        d/d cell = sum_T c_T ints_T (x) (vec + T) = (c @ ints) (x) vec + (sum_T c_T ints_T
        (x) ints_T) @ cell."""
    c = _direct_sum_coef(vec, cell, ints, gam, alpha, g)
    ci = c @ ints  # (P, 3)
    dvec = c.sum(1)[:, None] * vec + (ci[:, None, :] @ cell)[:, 0]
    outer = (ints[:, :, None] * ints[:, None, :]).reshape(-1, 9)  # (T, 9)
    dcell = ci[:, :, None] * vec[:, None, :] + (c @ outer).view(-1, 3, 3) @ cell
    return dvec, dcell


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


class _DirectCoulomb(torch.autograd.Function):
    """Direct-space part of the periodic Coulomb matrices of a batch, flat (S * n_max^2):
    w_e sum_T [erf(g r) - erf(alpha r)] / r per image pair e over the 5 x 5 x 5 block, at (i, j)
    and (j, i) of the pair's system, with an analytic backward. Both directions are reductions
    over the block evaluated in pair chunks (``compile``: fused, dynamic shapes), so no
    (pairs x 125) tensor is kept."""

    @staticmethod
    def forward(ctx, pos, cells, pairs, rad, alphas, local, n_max, compile):
        fn = _maybe_compiled(_direct_sum, compile)
        ints = _direct_block(cells.device, cells.dtype)
        cell = cells[pairs.system]  # (E, 3, 3)
        vec = pos[pairs.j] - pos[pairs.i] + (pairs.shift[:, None, :] @ cell)[:, 0]
        gam = torch.rsqrt(rad[pairs.i].square() + rad[pairs.j].square())
        alpha = alphas[pairs.system]
        values = torch.cat(
            [
                fn(vec[sl], cell[sl], ints, gam[sl], alpha[sl])
                for sl in (slice(s, s + _PAIR_CHUNK) for s in range(0, vec.shape[0], _PAIR_CHUNK))
            ]
        )
        base = pairs.system * (n_max * n_max)
        ij = base + local[pairs.i] * n_max + local[pairs.j]
        ji = base + local[pairs.j] * n_max + local[pairs.i]
        ctx.save_for_backward(vec, cell, gam, alpha, ij, ji)
        ctx.pairs, ctx.n_atoms, ctx.n_systems, ctx.compile = (
            pairs, pos.shape[0], cells.shape[0], compile
        )  # fmt: skip
        values = pairs.weight * values
        off = pairs.i != pairs.j
        out = values.new_zeros(cells.shape[0] * n_max * n_max)
        return out.index_add(0, ij, values).index_add(0, ji[off], values[off])

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_a,) = grad_outputs
        vec, cell, gam, alpha, ij, ji = ctx.saved_tensors
        pairs = ctx.pairs
        # d(sum grad_a * A) / d(entry e) = grad_a[i, j] + grad_a[j, i] off the diagonal
        off = pairs.i != pairs.j
        g = pairs.weight * torch.where(off, grad_a[ij] + grad_a[ji], grad_a[ij])
        fn = _maybe_compiled(_direct_sum_grad, ctx.compile)
        ints = _direct_block(cell.device, cell.dtype)
        dvecs, dcells = [], []
        for s in range(0, vec.shape[0], _PAIR_CHUNK):
            sl = slice(s, s + _PAIR_CHUNK)
            dv, dc = fn(vec[sl], cell[sl], ints, gam[sl], alpha[sl], g[sl])
            dvecs.append(dv)
            dcells.append(dc)
        dvec, dcell = torch.cat(dvecs), torch.cat(dcells)
        # vec = x_j - x_i + shift @ cell
        dcell = dcell + pairs.shift[:, :, None] * dvec[:, None, :]
        grad_pos = dvec.new_zeros(ctx.n_atoms, 3).index_add(0, pairs.j, dvec)
        grad_pos = grad_pos.index_add(0, pairs.i, -dvec)
        grad_cells = dcell.new_zeros(ctx.n_systems, 3, 3).index_add(0, pairs.system, dcell)
        return grad_pos, grad_cells, None, None, None, None, None, None


def direct_coulomb(
    pos: Tensor,
    cells: Tensor,
    rad: Tensor,
    alphas: Tensor,
    pairs: ImagePairs,
    batch: Batch,
    compile: bool,
) -> Tensor:
    """(S, n_max, n_max) direct-space part of the periodic EEQ Coulomb matrices; ``pos`` and
    ``rad`` per atom (N,), ``cells`` (S, 3, 3), ``alphas`` (S,). Differentiable in pos, cells."""
    flat = _DirectCoulomb.apply(pos, cells, pairs, rad, alphas, batch.local, batch.n_max, compile)
    return flat.view(batch.n_systems, batch.n_max, batch.n_max)


def _reciprocal_vectors(cells: Tensor) -> Tensor:
    """The 124 G vectors of the 5 x 5 x 5 reciprocal block (origin excluded), (S, 124, 3)."""
    ints = torch.tensor(_lattice_points((2, 2, 2), False), device=cells.device, dtype=cells.dtype)
    return ints @ (2.0 * torch.pi * torch.linalg.inv(cells).mT)


def coulomb_matrix_reciprocal(
    pos: Tensor, cells: Tensor, mask: Tensor, rad: Tensor, eta: Tensor, alphas: Tensor
) -> Tensor:
    """(S, n, n) reciprocal-space part plus the diagonal of the periodic EEQ Coulomb matrices
    (the direct part is ``direct_coulomb``). Padded layout: ``pos`` (S, n, 3), ``mask``, ``rad``,
    ``eta`` (S, n); empty rows get a decoupled unit diagonal. Free of host syncs."""
    g = _reciprocal_vectors(cells)
    g2 = g.square().sum(-1)  # (S, 124)
    volume = torch.linalg.det(cells).abs()
    a2 = (alphas * alphas)[:, None]
    f = (4.0 * torch.pi) * torch.exp(-0.25 * g2 / a2) / (g2 * volume[:, None])
    phase = pos @ g.mT  # (S, n, 124)
    m = mask[..., None].to(pos.dtype)
    cos, sin = torch.cos(phase) * m, torch.sin(phase) * m
    f = f[:, None, :]
    a_rec = (cos * f) @ cos.mT + (sin * f) @ sin.mT
    diag = eta + _SQRT2PI / rad - 2.0 * alphas[:, None] / math.sqrt(math.pi)
    return a_rec + torch.diag_embed(torch.where(mask, diag, 1.0))


def coulomb_matrix_molecular(pos: Tensor, mask: Tensor, rad: Tensor, eta: Tensor) -> Tensor:
    """(S, n, n) molecular EEQ Coulomb matrices in the padded layout (see the periodic one)."""
    n = pos.shape[1]
    vec = pos[:, None, :, :] - pos[:, :, None, :]
    eye = torch.eye(n, dtype=torch.bool, device=pos.device)
    off = ~eye & mask[:, :, None] & mask[:, None, :]
    r = torch.sqrt(torch.where(off, vec.square().sum(-1), 1.0))
    gam = torch.rsqrt(rad[:, :, None].square() + rad[:, None, :].square())
    a = torch.where(off, torch.erf(gam * r) / r, 0.0)
    return a + torch.diag_embed(torch.where(mask, eta + _SQRT2PI / rad, 1.0))


def solve_charges(a: Tensor, x: Tensor, mask: Tensor, total_charge: Tensor) -> Tensor:
    """q (S, n) from the bordered systems [A m; m^T 0] [q; lambda] = [x; Q], with m the real
    atoms (empty rows decouple with q = 0); ``x`` zero in empty rows, ``total_charge`` (S,)."""
    m = mask.to(a.dtype)
    top = torch.cat([a, m[..., None]], -1)
    bottom = torch.cat([m, m.new_zeros(m.shape[0], 1)], -1)[:, None, :]
    rhs = torch.cat([x, total_charge[:, None]], -1)
    return torch.linalg.solve(torch.cat([top, bottom], 1), rhs)[:, :-1]
