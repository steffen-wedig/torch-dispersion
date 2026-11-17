"""Configuration models (pydantic, yaml-loadable)."""

import json
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal

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
    """Fields shared by the real-space ATM settings and the experimental FourierATM settings."""

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


D4_FUNCTIONALS_PATH = Path(__file__).parent / "d4" / "functionals.json"
"""Frozen ``d4.bj-eeq-atm`` table of dftd4 4.3.0 (``scripts/export_d4_functionals.py``)."""


@cache
def _d4_functionals() -> dict[str, dict[str, float]]:
    return json.loads(D4_FUNCTIONALS_PATH.read_text())["functionals"]


class D4Settings(_EvalSettings):
    """DFT-D4 (dftd4 4.3.0 ``d4`` model, EEQ-2019 charges) with rational (Becke-Johnson)
    damping: two-body term plus ATM three-body term. The damping parameters are the functional's
    ``d4.bj-eeq-atm`` entry of dftd4's ``parameters.toml`` (e.g. r2SCAN: s8 = 0.60187490,
    a1 = 0.51559235, a2 = 5.77342911); ``from_functional`` looks them up."""

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
    pair_block_size: int = Field(default=512, gt=0)
    """Centres per two-body kernel call. Much larger than the ATM's ``centre_block_size``: a
    pair tile is (B, T), not (B, T, T), and fewer, larger calls keep the per-call overhead
    small."""
    pair_tile_size: int = Field(default=4096, gt=0)
    """Neighbour tile width of the two-body kernel (``pair_block_size`` x ``pair_tile_size``
    pairs per call)."""
    min_neighbor_tile_size: int = Field(default=32, gt=0)
    """Narrowest ATM tile. Each centre block gets the smallest power-of-two multiple of this
    that holds its longest neighbour list, at most ``neighbor_tile_size``, so blocks of small
    molecules do not pay for 512-wide tiles. Full-width tiles run a compiled graph with a static
    width, and all narrower widths share one graph with a dynamic width. Set it equal to
    ``neighbor_tile_size`` for one tile width everywhere."""

    @classmethod
    def from_functional(cls, name: str, atm: bool = True, **overrides: Any) -> "D4Settings":
        """Settings with the damping parameters of functional ``name`` (case-insensitive,
        otherwise exact: ``r2scan-3c``, ``pbe0_2``). The entry's parameters (s6, s8, s9, a1,
        a2, alp, merged onto dftd4's defaults) come first, then ``atm=False`` sets s9 = 0 (no
        three-body term), then ``overrides`` (any field: device, compile, cutoffs, ...)."""
        table = _d4_functionals()
        key = name.lower()
        if key not in table:
            raise ValueError(
                f"unknown D4 functional {name!r}; available: {', '.join(sorted(table))}"
            )
        params: dict[str, Any] = dict(table[key])
        if not atm:
            params["s9"] = 0.0
        return cls(**{**params, **overrides})

    @staticmethod
    def functionals() -> list[str]:
        """Names accepted by ``from_functional`` (lower case, sorted)."""
        return sorted(_d4_functionals())
