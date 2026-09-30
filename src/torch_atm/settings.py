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


class ATMSettings(BaseModel):
    """Parameters of the ATM term and of its evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    s9: float = 1.0
    alpha: float = 14.0
    """The damping set's ``alp``; the ATM damping uses ``alpha + 2``."""
    rs9: float = 4.0 / 3.0
    three_body_cutoff_bohr: float = Field(default=40.0, gt=0)
    three_body_cutoff_width_bohr: float = Field(default=0.0, ge=0)
    coordination_cutoff_bohr: float = Field(default=40.0, gt=0)
    centre_block_size: int = Field(default=16, gt=0)
    """Centres per kernel call."""
    neighbor_tile_size: int = Field(default=256, gt=0)
    """Neighbour tile width of the kernel. One call evaluates ``centre_block_size`` centres times
    a ``neighbor_tile_size`` x ``neighbor_tile_size`` tile of neighbour pairs, so this bounds the
    kernel memory independently of the neighbour count. It is also the padding bucket of the
    neighbour axis."""
    neighbor_backend: NeighborBackend = BruteForceNeighbors()
    bohr_in_angstrom: float = Field(default=ase.units.Bohr, gt=0)
    hartree_in_ev: float = Field(default=ase.units.Hartree, gt=0)
    dtype: Literal["float64", "float32"] = "float64"
    device: str = "cuda"
    compile: bool = True

    @model_validator(mode="after")
    def _check_width(self) -> "ATMSettings":
        if self.three_body_cutoff_width_bohr >= self.three_body_cutoff_bohr:
            raise ValueError("three_body_cutoff_width_bohr must be smaller than the cutoff")
        return self

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)
