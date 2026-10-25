"""TEST-ONLY D3(BJ) two-body energy. It exists to validate the CN -> weights -> C6 chain
against the oracle (design §8, T1) and is not a production two-body term.

E_2 = -1/2 sum_{i, (j, T)} C6_ij [s6 / (r^6 + R^6) + s8 rr_ij / (r^8 + R^8)],
rr_ij = 3 r4r2_i r4r2_j, R = a1 sqrt(rr_ij) + a2 (simple-dftd3 ``rational.f90``).
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from .atm import CN_WIDTH_MULTIPLE, check_pbc, pad_positions, pad_system, stage1_fn
from .data import check_numbers, reference_data
from .neighbors import make_builder
from .settings import ATMSettings


def two_body_energy(
    numbers: Tensor,
    positions: Tensor,
    cell: Tensor | None,
    pbc: bool | Sequence[bool] | Tensor,
    settings: ATMSettings,
    s6: float,
    s8: float,
    a1: float,
    a2: float,
    cutoff_bohr: float = 60.0,
) -> Tensor:
    """Two-body D3(BJ) energy in eV (float64)."""
    pbc = check_pbc(pbc)
    device, dtype = settings.torch_device, settings.torch_dtype
    numbers = torch.as_tensor(numbers).to(device=device, dtype=torch.int64)
    n = numbers.shape[0]
    data = reference_data(device, dtype)
    check_numbers(numbers, data)
    bohr = settings.bohr_in_angstrom
    pos = torch.as_tensor(positions).to(device=device, dtype=dtype) / bohr
    cell_b = torch.as_tensor(cell).to(device=device, dtype=dtype) / bohr if pbc else None

    builder = make_builder(settings.neighbor_backend)
    ccn = settings.coordination_cutoff_bohr
    n_padded = n + 1
    nl_cn = builder(pos, cell_b, pbc, ccn).pad(n_padded, CN_WIDTH_MULTIPLE)
    system = pad_system(numbers, n_padded, data)
    _, c6 = stage1_fn(False)(
        pad_positions(pos, n_padded),
        cell_b,
        nl_cn,
        system.rcov,
        system.ref_cn,
        system.ref_mask,
        system.species,
        system.c6_species,
        ccn,
    )

    nl = builder(pos, cell_b, pbc, cutoff_bohr)
    i, j, shifts = nl.to_pairs()
    vec = pos[j] - pos[i]
    if cell_b is not None:
        vec = vec + shifts.to(dtype) @ cell_b
    r2 = vec.square().sum(-1)
    keep = r2 > torch.finfo(dtype).eps
    i, j, r2 = i[keep], j[keep], r2[keep]
    rr = 3.0 * data.r4r2[numbers[i]] * data.r4r2[numbers[j]]
    r0 = a1 * torch.sqrt(rr) + a2
    t6 = 1.0 / (r2**3 + r0**6)
    t8 = 1.0 / (r2**4 + r0**8)
    e = -0.5 * c6[i, j] * (s6 * t6 + s8 * rr * t8)
    return e.to(torch.float64).sum() * settings.hartree_in_ev
