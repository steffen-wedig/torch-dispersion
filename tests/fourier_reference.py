"""Brute-force reference for FourierATM: the real-space triangle sum (torch-dispersion's tiles) of

    e = k_D3 fdmp ang S + k_CP ang (1 - S),    S = s(r_ij) s(r_ik) s(r_jk) at r_s,

over all triangles with legs <= ``cutoff``. As the cutoff grows this converges (as cutoff^-3)
to the FourierATM energy with the same separable C9."""

import torch

import torch_dispersion.atm.kernel as kernel_module
from torch_dispersion.atm import atm_dispersion
from torch_dispersion.atm.fourier import separable_c9
from torch_dispersion.settings import ATMSettings, FourierATMSettings


def reference_energy(
    atoms, fs: FourierATMSettings, cutoff: float, device="cpu", compile=False
) -> float:
    """eV. Uses the ATM real-space machinery at ``cutoff`` with a patched per-triangle energy."""
    pbc = bool(atoms.pbc.all())
    cell = torch.tensor(atoms.cell.array) if pbc else None
    a, mu = separable_c9(atoms.numbers, atoms.positions, cell, pbc, fs)
    n = len(atoms)
    orig = kernel_module._terms
    r_s = torch.tensor(fs.short_range_cutoff_bohr, dtype=torch.float64)
    width = torch.tensor(fs.short_range_width_bohr, dtype=torch.float64)

    def patched(v_n, v_m, c6, t, s):
        x = orig(v_n, v_m, c6, t, s)
        dt = x.u_ij.dtype
        a_pad = torch.cat([a, a.new_zeros(c6.shape[0] - n, a.shape[1])]).to(dt)
        k_cp = torch.bmm(
            a_pad[t.idx_n] * (a_pad[t.centre] * mu.to(dt))[:, None, :],
            a_pad[t.idx_m].transpose(1, 2),
        )
        k_cp = (s.s9 / 3.0) * k_cp
        sw = [
            kernel_module._switch(u, r_s.to(dt), width.to(dt))[0] for u in (x.u_ij, x.u_ik, x.u_jk)
        ]
        big_s = sw[0][:, :, None] * sw[1][:, None, :] * sw[2]
        e = torch.where(x.valid, (x.k * x.fdmp * big_s + k_cp * (1.0 - big_s)) * x.ang, 0.0)
        return x._replace(e=e)

    settings = ATMSettings(
        device=device,
        compile=compile,
        s9=fs.s9,
        alpha=fs.alpha,
        rs9=fs.rs9,
        three_body_cutoff_bohr=cutoff,
        coordination_cutoff_bohr=fs.cn_cutoff,
        bohr_in_angstrom=fs.bohr_in_angstrom,
        hartree_in_ev=fs.hartree_in_ev,
    )
    if fs.coordination != "d3":
        raise ValueError("the reference uses the D3 CN")
    kernel_module._terms = patched
    try:
        r = atm_dispersion(
            atoms.numbers,
            atoms.positions,
            cell,
            pbc,
            settings,
            compute_forces=False,
            compute_stress=False,
        )
    finally:
        kernel_module._terms = orig
    return float(r.energy)
