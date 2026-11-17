"""Settings of the experimental cutoff-free ATM (``FourierATMSettings``)."""

from typing import Literal

import ase.units
from pydantic import Field, model_validator

from ...settings import _KernelSettings

SMOOTH_CN_CUTOFF_BOHR = 6.0 / ase.units.Bohr
"""FourierD3's default ``smooth_cut`` CN range, 6 Å."""


class FourierATMSettings(_KernelSettings):
    """Cutoff-free ATM: dense triple-dipole trace (undamped, separable C9) plus a real-space
    short-range correction to exact D3 (see ``torch_dispersion.experimental.fourier_atm``)."""

    coordination: Literal["smooth_cut", "d3"] = "smooth_cut"
    """``smooth_cut``: FourierD3's CN (shared with its two-body term); ``d3``: the exact D3 CN
    (exp counting, hard cutoff), as in simple-dftd3 and the real-space ATM."""
    coordination_cutoff_bohr: float | None = Field(default=None, gt=0)
    """None: 6 Å for ``smooth_cut``, 40 bohr for ``d3``."""
    short_range_cutoff_bohr: float = Field(default=20.0, gt=0)
    """Triangles with all legs inside get exact D3 damping; outside they are undamped (error
    <= 0.002 meV/atom on the fixture liquids at 20 bohr, 0.08 meV/atom at 15 bohr)."""
    short_range_width_bohr: float = Field(default=3.0, ge=0)
    """Width of the quintic switch between the two regimes."""
    cp_rank: int = Field(default=2, gt=0)
    """Terms of the separable C9. Rank 2 reaches a max relative error of 2e-3 (H C N O) to 4e-3
    (H C N O F S Br) on the reference tensor; higher ranks barely improve it."""
    ewald_real_cutoff_bohr: float = Field(default=30.0, gt=0)
    """Real-space range of the Ewald dipole sum. Longer means fewer k-vectors: on the 250-atom
    liquids (H200) 20 / 25 / 30 bohr take 0.076 / 0.058 / 0.052 s for energy, forces, stress."""
    ewald_accuracy: float = Field(default=1e-10, gt=0, lt=1)
    """Sets the splitting g = sqrt(-ln acc) / real cutoff and the k cutoff 2 g sqrt(-ln acc)."""
    trace_gradient: Literal["analytic", "autograd"] = "analytic"
    """Gradients of the periodic trace: hand-written (``fourier/trace_grad.py``; the q -> 0
    correction stays on autograd) or autograd (the reference; they agree to ~1e-14). Analytic is
    1.3-1.6x faster with 2-2.5x less peak memory from 512 atoms (H200). Molecules always use
    autograd."""
    q_grid: tuple[int, int, int] | None = None
    """Bloch grid; None: the smallest grid whose supercell is ``supercell_length_bohr`` wide."""
    supercell_length_bohr: float = Field(default=70.0, gt=0)

    @model_validator(mode="after")
    def _check_width(self) -> "FourierATMSettings":
        if self.short_range_width_bohr >= self.short_range_cutoff_bohr:
            raise ValueError("short_range_width_bohr must be smaller than the cutoff")
        return self

    @property
    def cn_cutoff(self) -> float:
        if self.coordination_cutoff_bohr is not None:
            return self.coordination_cutoff_bohr
        return SMOOTH_CN_CUTOFF_BOHR if self.coordination == "smooth_cut" else 40.0
