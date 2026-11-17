"""FourierATM (experimental): cutoff-free D3 ATM from a dense triple-dipole trace plus a real-space
short-range correction (docs/experimental/fourier_atm.md)."""

from .ase import FourierATMCalculator
from .model import FourierATMResult, fourier_atm, long_range_energy, separable_c9
from .settings import FourierATMSettings

__all__ = [
    "FourierATMCalculator",
    "FourierATMResult",
    "FourierATMSettings",
    "fourier_atm",
    "long_range_energy",
    "separable_c9",
]
