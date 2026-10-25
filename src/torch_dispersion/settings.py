"""Configuration models (pydantic, yaml-loadable)."""

from typing import Annotated, Literal

import ase.units
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BruteForceNeighbors(BaseModel):
    """All pairs times all lattice translations, evaluated in torch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["brute_force"] = "brute_force"
    pair_block_size: int = Field(default=4096, gt=0)
    """Atom pairs (i, j) per block of the scan; each pair is evaluated for all translations,
    so a block holds ``pair_block_size * n_translations`` distances."""


class NvalchemiNeighbors(BaseModel):
    """``nvalchemiops.torch.neighbors.neighbor_list`` (warp)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["nvalchemi"] = "nvalchemi"
    method: Literal["naive", "cell_list"] | None = None
    """None lets the library choose."""
    max_neighbors: int | None = Field(default=None, gt=0)
    """None estimates the width from the density with headroom. Overflow is always an error."""


NeighborBackend = Annotated[BruteForceNeighbors | NvalchemiNeighbors, Field(discriminator="kind")]


class _EvalSettings(BaseModel):
    """Evaluation fields shared by every term (kernel shape, backends, units, device)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    centre_block_size: int = Field(default=32, gt=0)
    """Centres per kernel call. Defaults (32, 512) from the H200 sweep: 0.30 s for energy, forces
    and stress of a 250-atom liquid at 40 bohr (B = 16: 0.34 s, B = 64: 0.28 s at 1.6x memory);
    smaller shapes are bound by per-call overhead."""
    neighbor_tile_size: int = Field(default=512, gt=0)
    """Neighbour tile width of the kernel. One call evaluates ``centre_block_size`` centres times
    a ``neighbor_tile_size`` x ``neighbor_tile_size`` tile of neighbour pairs, so this bounds the
    kernel memory independently of the neighbour count. It is also the padding bucket of the
    neighbour axis."""
    kernel_gradient: Literal["analytic", "vjp"] = "analytic"
    """Triple-kernel derivatives: hand-written, or ``torch.func.vjp`` (the reference)."""
    neighbor_backend: NeighborBackend = BruteForceNeighbors()
    bohr_in_angstrom: float = Field(default=ase.units.Bohr, gt=0)
    hartree_in_ev: float = Field(default=ase.units.Hartree, gt=0)
    dtype: Literal["float64", "float32"] = "float64"
    device: str = "cuda"
    compile: bool = True

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


class _KernelSettings(_EvalSettings):
    """Fields shared by the real-space ATM and FourierATM settings."""

    s9: float = 1.0
    alpha: float = 14.0
    """The damping set's ``alp``; the ATM damping uses ``alpha + 2``."""
    rs9: float = 4.0 / 3.0


class ATMSettings(_KernelSettings):
    """Parameters of the ATM term and of its evaluation."""

    three_body_cutoff_bohr: float = Field(default=40.0, gt=0)
    three_body_cutoff_width_bohr: float = Field(default=0.0, ge=0)
    coordination_cutoff_bohr: float = Field(default=40.0, gt=0)

    @model_validator(mode="after")
    def _check_width(self) -> "ATMSettings":
        if self.three_body_cutoff_width_bohr >= self.three_body_cutoff_bohr:
            raise ValueError("three_body_cutoff_width_bohr must be smaller than the cutoff")
        return self


SMOOTH_CN_CUTOFF_BOHR = 6.0 / ase.units.Bohr
"""FourierD3's default ``smooth_cut`` CN range, 6 Å."""


class FourierATMSettings(_KernelSettings):
    """Cutoff-free ATM: dense triple-dipole trace (undamped, separable C9) plus a real-space
    short-range correction to exact D3 (see ``torch_dispersion.fourier``)."""

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


class D4Settings(_EvalSettings):
    """DFT-D4 (dftd4 4.3.0 ``d4`` model, EEQ-2019 charges) with rational (Becke-Johnson)
    damping: two-body term plus ATM three-body term. The damping parameters are the functional's
    ``d4.bj-eeq-atm`` entry of dftd4's ``parameters.toml`` (e.g. r2SCAN: s8 = 0.60187490,
    a1 = 0.51559235, a2 = 5.77342911)."""

    s6: float = 1.0
    s8: float
    s9: float = 1.0
    a1: float
    a2: float
    alp: float = 16.0
    """ATM damping exponent (used as is, unlike D3's ``alpha + 2``)."""
    two_body_cutoff_bohr: float = Field(default=60.0, gt=0)
    three_body_cutoff_bohr: float = Field(default=40.0, gt=0)
    coordination_cutoff_bohr: float = Field(default=30.0, gt=0)
    """Range of the D4 coordination number. The EEQ coordination number has its own fixed
    25 bohr range (part of the EEQ-2019 model)."""
    pair_tile_size: int = Field(default=4096, gt=0)
    """Neighbour tile width of the two-body kernel (``centre_block_size`` x ``pair_tile_size``
    pairs per call); also the padding bucket of its neighbour axis."""
