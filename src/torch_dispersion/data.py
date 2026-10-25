"""D3 reference tables as tensors, cached per (device, dtype).

All lengths in bohr, C6 in Hartree bohr^6. The tables are exported by
``scripts/export_reference_data.py`` from tad-dftd3 and tad-mctc.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

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


def check_numbers(numbers: Tensor, data: ReferenceData) -> None:
    if numbers.numel() and (int(numbers.min()) < 1 or int(numbers.max()) > data.max_number):
        raise ValueError(f"atomic numbers must be in 1..{data.max_number}")
