"""The D3 Axilrod-Teller-Muto three-body term (simple-dftd3 1.6.0) in real space with a cutoff
(``realspace``), on the per-system triple kernel (``kernel``), and the batch-wide triple kernel
used by D4 (``flat``). The cutoff-free variant is ``torch_dispersion.experimental.fourier_atm``."""

from .realspace import ATMResult, atm_dispersion

__all__ = ["ATMResult", "atm_dispersion"]
