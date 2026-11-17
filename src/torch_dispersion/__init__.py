"""Dispersion corrections in PyTorch, with energy, forces and stress for molecules and 3D cells.

Import from the submodules:

- ``torch_dispersion.d4``: ``d4_dispersion``, ``d4_dispersion_batched`` (DFT-D4, dftd4 4.3.0,
  including ORCA's D4rev parameter sets);
- ``torch_dispersion.atm``: ``atm_dispersion`` (D3 Axilrod-Teller-Muto term, simple-dftd3 1.6.0);
- ``torch_dispersion.settings``: ``D4Settings``, ``ATMSettings``, ``NvalchemiNeighbors``,
  ``BruteForceNeighbors``;
- ``torch_dispersion.calculators.ase``: ``TorchD4Calculator``, ``TorchATMCalculator``;
- ``torch_dispersion.calculators.torchsim``: ``D4Model`` (needs the ``torchsim`` extra);
- ``torch_dispersion.experimental``: methods whose API may still change (``fourier_atm``).
"""

from importlib.metadata import version

__version__ = version("torch-dispersion")
