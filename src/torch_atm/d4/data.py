"""D4 reference tables as tensors, cached per (device, dtype).

Lengths in bohr, polarisabilities in bohr^3 on dftd4's 23-point imaginary frequency grid. The
tables are exported by ``scripts/export_d4_reference_data.py`` (tad-dftd4, tad-multicharge,
tad-mctc) for Z = 0..103; row 0 is the padding dummy.
"""

import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

_PATH = Path(__file__).with_name("reference_data.npz")

MAX_REFERENCES = 7

# imaginary frequencies of the Casimir-Polder grid (dftd4 ``trapzd``)
FREQUENCIES = (
    0.000001, 0.050000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000,
    0.700000, 0.800000, 0.900000, 1.000000, 1.200000, 1.400000, 1.600000, 1.800000,
    2.000000, 2.500000, 3.000000, 4.000000, 5.000000, 7.500000, 10.00000,
)  # fmt: skip


def _trapezoid_weights() -> list[float]:
    f = FREQUENCIES
    gaps = [f[k + 1] - f[k] for k in range(len(f) - 1)]
    return [0.5 * (a + b) for a, b in zip([0.0, *gaps], [*gaps, 0.0], strict=True)]


TRAPEZOID_WEIGHTS = tuple(_trapezoid_weights())


@dataclass(frozen=True)
class D4ReferenceData:
    covalent_radius: Tensor  # (Z,) D3 covalent radii (4/3 scaled)
    electronegativity: Tensor  # (Z,) Pauling
    zeff: Tensor  # (Z,) effective nuclear charges
    hardness: Tensor  # (Z,) chemical hardness of the charge scaling
    r4r2: Tensor  # (Z,) sqrt(0.5 <r4>/<r2> sqrt(Z))
    n_ref: Tensor  # (Z,) int64
    n_gaussians: Tensor  # (Z, 7) int64, Gaussians per reference (0 for missing references)
    reference_cn: Tensor  # (Z, 7)
    reference_q: Tensor  # (Z, 7)
    alpha_iw: Tensor  # (Z, 7, 23) reference polarisabilities
    eeq_chi: Tensor  # (Z,) EEQ-2019 electronegativity
    eeq_eta: Tensor  # (Z,) chemical hardness
    eeq_kcnchi: Tensor  # (Z,) CN dependence of the electronegativity
    eeq_rad: Tensor  # (Z,) charge widths

    @property
    def max_number(self) -> int:
        return self.n_ref.shape[0] - 1


_INTEGER = ("n_ref", "n_gaussians")


@cache
def _raw() -> dict[str, Tensor]:
    with np.load(_PATH) as tables:
        return {k: torch.from_numpy(tables[k]) for k in tables.files}


@cache
def d4_reference_data(device: torch.device, dtype: torch.dtype) -> D4ReferenceData:
    return D4ReferenceData(
        **{
            k: v.to(device=device, dtype=v.dtype if k in _INTEGER else dtype)
            for k, v in _raw().items()
        }
    )


def reference_c6(alpha_a: Tensor, alpha_b: Tensor) -> Tensor:
    """Casimir-Polder C6 of reference pairs, 3/pi int alpha_a(iw) alpha_b(iw) dw (trapezoid).

    alpha_a (S, R, 23), alpha_b (S', R', 23) -> (S, S', R, R').
    """
    w = alpha_a.new_tensor(TRAPEZOID_WEIGHTS) * (3.0 / math.pi)
    return torch.einsum("sak,tbk,k->stab", alpha_a, alpha_b, w)
