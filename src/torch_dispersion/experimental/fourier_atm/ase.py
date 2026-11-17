"""ASE calculator of the experimental FourierATM term."""

from ...calculators.ase import ThreeBodyCalculator
from .model import fourier_atm
from .settings import FourierATMSettings


class FourierATMCalculator(ThreeBodyCalculator):
    """The cutoff-free FourierATM term as an ASE calculator (energy, forces, stress)."""

    def __init__(self, settings: FourierATMSettings | None = None, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings if settings is not None else FourierATMSettings()

    def _evaluate(
        self, numbers, positions, cell, pbc: bool, compute_forces: bool, compute_stress: bool
    ):
        return fourier_atm(
            numbers, positions, cell, pbc, self.settings, compute_forces, compute_stress
        )
