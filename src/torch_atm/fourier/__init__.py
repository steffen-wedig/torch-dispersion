"""FourierATM: cutoff-free D3 ATM from a dense triple-dipole trace plus a short-range
correction (plan: docs/fourier_atm_plan.md)."""

from .model import FourierATMResult, fourier_atm, long_range_energy, separable_c9

__all__ = ["FourierATMResult", "fourier_atm", "long_range_energy", "separable_c9"]
