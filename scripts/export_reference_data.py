"""Export the D3 reference tables into ``src/torch_dispersion/reference_data.pt``.

Sources (Apache-2.0): tad-dftd3 (reference CNs, reference C6, r4/r2 expectation values) and
tad-mctc (D3 covalent radii, pairwise zero-damping vdW radii). All lengths are in bohr.

The length tables are rebuilt from their Angstrom values with mctc-lib's conversion factor
(CODATA 2018, ``1 / (hbar / (m_e c alpha) * 1e10)``). tad-mctc's ``vdw-pairwise.pt`` is stored
with ~1e-7 relative noise, which shifts the ATM damping (it goes as R0^(16/3)) by the same
order; the published radii have 4 (R0) and 2 (covalent) decimals in Angstrom, so rounding the
tad values in Angstrom recovers them exactly (verified against simple-dftd3's ``vdwrad.f90``).

``r4r2`` is recomputed in float64 from the raw ``r4_over_r2`` values: tad-dftd3 stores those
and evaluates ``r4r2`` in torch's default dtype (float32), ~1e-8 relative noise. The raw values
have 4 decimals, so rounding the float32 values recovers them exactly.

Run with ``uv run python scripts/export_reference_data.py``; the checksum in
``tests/test_data.py`` must be updated whenever the output changes.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from tad_dftd3.data.r4r2 import r4_over_r2
from tad_dftd3.reference import _load_c6, _load_cn
from tad_mctc.data.radii import COV_D3, VDW_PAIRWISE
from tad_mctc.units.length import AA2AU as TAD_AA2AU

# mctc-lib codata2018: h, m_e, c, alpha
_HBAR = 6.62607015e-34 / (2.0 * math.pi)
_BOHR_M = _HBAR / (9.1093837015e-31 * 299792458.0 * 7.2973525693e-3)
AATOAU = 1.0 / (_BOHR_M * 1e10)


def _exact_bohr(table: torch.Tensor, decimals: int, scale: float = 1.0) -> torch.Tensor:
    angstrom = torch.round(table / (TAD_AA2AU * scale), decimals=decimals)
    return AATOAU * scale * angstrom


OUT = Path(__file__).resolve().parents[1] / "src" / "torch_dispersion" / "reference_data.npz"


def main() -> None:
    dtype = torch.float64
    ref_cn = _load_cn(dtype=dtype)  # (Z, 7), -1 marks a missing reference
    c6_ref = _load_c6(dtype=dtype)  # (Z, Z, 7, 7)
    n_ref = (ref_cn >= 0).sum(-1).to(torch.int64)
    raw = torch.round(r4_over_r2.to(dtype), decimals=4)
    r4r2 = torch.sqrt(0.5 * raw * torch.sqrt(torch.arange(raw.shape[0], dtype=dtype)))
    tables = {
        "covalent_radius": _exact_bohr(COV_D3(dtype=dtype), 2, scale=4.0 / 3.0),
        "reference_cn": ref_cn,
        "n_ref": n_ref,
        "c6_ref": c6_ref,
        "r0_vdw": _exact_bohr(VDW_PAIRWISE(dtype=dtype), 4),
        "r4r2": r4r2,
    }
    # the C6 reference covers Z = 0..103; truncate the element tables to the same range
    n_z = ref_cn.shape[0]
    tables["covalent_radius"] = tables["covalent_radius"][:n_z].contiguous()
    tables["r4r2"] = tables["r4r2"][:n_z].contiguous()
    for name, t in tables.items():
        assert t.shape[0] == n_z, (name, t.shape)
    # compressed: the dense (Z, Z, 7, 7) C6 block is mostly structural zeros
    np.savez_compressed(OUT, **{k: v.numpy() for k, v in tables.items()})
    print(f"wrote {OUT}")
    print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
