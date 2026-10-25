"""D3 reference tables as tensors, cached per (device, dtype), and the padded per-atom tables.

All lengths in bohr, C6 in Hartree bohr^6. The tables are exported by
``scripts/export_reference_data.py`` from tad-dftd3 and tad-mctc.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..core.padding import pad_numbers

_PATH = Path(__file__).with_name("reference_data.npz")

MAX_REFERENCES = 7


@dataclass(frozen=True)
class ReferenceData:
    covalent_radius: Tensor  # (Z,)
    reference_cn: Tensor  # (Z, 7), -1 for missing references
    n_ref: Tensor  # (Z,) int64
    c6_ref: Tensor  # (Z, Z, 7, 7)
    r0_vdw: Tensor  # (Z, Z) zero-damping pairwise cutoff radii
    r4r2: Tensor  # (Z,) sqrt(0.5 * <r4>/<r2> * sqrt(Z))

    @property
    def max_number(self) -> int:
        return self.reference_cn.shape[0] - 1


@cache
def _raw() -> dict[str, Tensor]:
    with np.load(_PATH) as tables:
        return {k: torch.from_numpy(tables[k]) for k in tables.files}


@cache
def reference_data(device: torch.device, dtype: torch.dtype) -> ReferenceData:
    raw = _raw()
    return ReferenceData(
        **{k: v.to(device=device, dtype=v.dtype if k == "n_ref" else dtype) for k, v in raw.items()}
    )


@dataclass
class D3System:
    """Atoms padded with dummies (Z = 0, at the origin, no references) to ``n_padded`` rows."""

    n: int
    numbers: Tensor  # (Np,)
    rcov: Tensor  # (Np,)
    ref_cn: Tensor  # (Np, R)
    ref_mask: Tensor  # (Np, R)
    species: Tensor  # (Np,)
    c6_species: Tensor  # (S, S, R, R)


def pad_system(numbers: Tensor, n_padded: int, data: ReferenceData) -> D3System:
    z, unique, species = pad_numbers(numbers, n_padded)
    ref_mask = torch.arange(MAX_REFERENCES, device=z.device) < data.n_ref[z][:, None]
    return D3System(
        n=numbers.shape[0],
        numbers=z,
        rcov=data.covalent_radius[z],
        ref_cn=data.reference_cn[z],
        ref_mask=ref_mask,
        species=species,
        c6_species=data.c6_ref[unique][:, unique],
    )
