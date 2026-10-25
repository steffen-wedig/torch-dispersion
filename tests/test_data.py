"""Reference tables: checksum and structural sanity (design §5.2)."""

import hashlib

import torch

from torch_dispersion.data import _raw, reference_data

# update together with scripts/export_reference_data.py
CHECKSUM = "bbbcae3ff145c6ddd574eac17cc9fa97"


def table_checksum() -> str:
    h = hashlib.sha256()
    for key, value in sorted(_raw().items()):
        h.update(key.encode())
        h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()[:32]


def test_checksum():
    assert table_checksum() == CHECKSUM


def test_table_structure():
    d = reference_data(torch.device("cpu"), torch.float64)
    n_z = d.max_number + 1
    assert d.c6_ref.shape == (n_z, n_z, 7, 7)
    torch.testing.assert_close(d.c6_ref, d.c6_ref.permute(1, 0, 3, 2), rtol=0, atol=0)
    torch.testing.assert_close(d.r0_vdw, d.r0_vdw.T, rtol=0, atol=0)
    assert int(d.n_ref[1]) == 2 and int(d.n_ref[6]) == 5
    # exact values of the source tables (Angstrom * mctc-lib aatoau, 4/3 for covalent radii)
    aatoau = 1.8897261246204404
    assert float(d.r0_vdw[1, 1]) == 2.1823 * aatoau
    assert float(d.covalent_radius[6]) == aatoau * 4.0 / 3.0 * 0.75


def test_float32_cache_is_separate():
    d64 = reference_data(torch.device("cpu"), torch.float64)
    d32 = reference_data(torch.device("cpu"), torch.float32)
    assert d32.c6_ref.dtype == torch.float32 and d64.c6_ref.dtype == torch.float64
    assert d32.n_ref.dtype == torch.int64
