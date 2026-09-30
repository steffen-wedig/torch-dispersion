# torch-atm

A GPU-capable, `torch.compile`-able PyTorch implementation of the D3
**Axilrod–Teller–Muto (ATM) three-body dispersion term**. It reproduces simple-dftd3 1.6.0
(energy, forces, stress) for molecules and fully periodic cells, including the CN-dependent C6
coefficients the term is built on. The specification is [`docs/design.md`](docs/design.md).

## Installation

```bash
uv sync                      # core: torch, ase, numpy, pydantic
uv sync --extra nvalchemi    # + the nvalchemiops (warp) neighbour-list backend
```

`torch.compile` on **CPU** needs a C++20 compiler (inductor); on DAIS run `module load gcc/14`
and `export CXX=$(which g++)` first. CUDA compilation uses Triton and needs nothing extra.

## Usage

```python
import torch
from torch_atm import ATMSettings, atm_dispersion

settings = ATMSettings(device="cuda")  # s9 = 1, alpha = 14, cutoffs 40 bohr
result = atm_dispersion(numbers, positions, cell, pbc=True, settings=settings)
result.energy, result.forces, result.stress  # eV, eV/Å, eV/Å³ (ASE sign convention)
```

With ASE, e.g. summed with an MLIP and a two-body D3:

```python
from ase.calculators.mixing import SumCalculator
from torch_atm import TorchATMCalculator

atoms.calc = SumCalculator([mlip, two_body_d3, TorchATMCalculator(settings)])
```

Settings are a pydantic model and load from YAML/JSON. To match FHI-aims numbers bit for bit,
pass the aims unit constants:

```python
ATMSettings(bohr_in_angstrom=0.52917721, hartree_in_ev=27.211384500)
```

Main options:

| setting | default | meaning |
|---|---|---|
| `s9`, `alpha`, `rs9` | 1.0, 14.0, 4/3 | ATM scale, damping-set `alp` (the ATM uses `alpha + 2`), radius scale |
| `three_body_cutoff_bohr` | 40 | hard cutoff on all three legs |
| `three_body_cutoff_width_bohr` | 0 | > 0 multiplies each leg by a quintic switch on (cutoff − width, cutoff); use it for MD with a shortened cutoff |
| `coordination_cutoff_bohr` | 40 | hard CN cutoff |
| `neighbor_backend` | brute force | `{"kind": "brute_force"}` or `{"kind": "nvalchemi", "max_neighbors": ...}` |
| `centre_block_size`, `neighbor_tile_size` | 16, 256 | kernel shape: B centres × T × T neighbour pairs per call |
| `dtype`, `device`, `compile` | float64, cuda, True | |

## How it works

1. **Graph stage** (autograd): positions and a zero strain → CN → Gaussian reference weights
   → C6 matrix (N × N).
2. **Triple stage** (blocked): C6 is detached; a Python loop over blocks of B centres and pairs
   of T-wide tiles of their neighbour lists calls a fixed-shape kernel that returns the tile
   energy and, through `torch.func.vjp`, its gradients with respect to the tile's displacement
   vectors and to C6. These are scatter-accumulated into dE/dR, the strain virial and dE/dC6, so
   memory is bounded by one tile, not by the ~10⁹ triangles of a liquid box.
3. **Chain closure**: `torch.autograd.grad(C6, ..., grad_outputs=dE/dC6)` adds the CN part.

Every unordered neighbour pair {n, m} of every centre i is visited once and the sum is divided
by 3, which covers triangles through periodic images of the same atom (the one-atom cell case).

**Recompilation control.** The kernel's heavy tensors are (B, T) and (B, T, T), fixed by the
settings; its only system-sized input (the C6 matrix) has dynamic dimensions and its parameters
are 0-d tensors. So one kernel graph serves every system, cutoff and neighbour count; only a new
`(centre_block_size, neighbor_tile_size)`, dtype or device compiles a new one. Stage 1 (CN, C6)
is compiled with dynamic shapes. MD steps do not recompile (tested).

## Tests

```bash
uv run pytest -m "not slow and not gpu"   # fast CPU suite
uv run pytest -m "not gpu"                # + slow CPU tests (liquid boxes: long on CPU)
sbatch scripts/slurm/gpu_tests.sbatch     # GPU tests incl. slow, on one H200/B200 (DAIS)
```

The oracle is `dftd3==1.6.0` (a dev dependency). Tests compare against it live and against the
stored fixtures in `tests/fixtures/`, which use the FHI-aims unit constants.

## Pitfalls (design §11, and what was found while implementing)

- The ATM exponent is `alp + 2`; `rs9 = 4/3`; r0 and r1 are **products** over the three legs.
- R0 is the zero-damping pairwise radius table, not the BJ radius.
- C6 is the full CN-dependent D3 C6 with a 40 bohr hard CN cutoff and exp counting (k1 = 16).
- **Reference-table precision.** tad-mctc's `vdw-pairwise.pt` carries ~1e-7 relative noise and
  tad-dftd3 builds `r4r2` in float32. Through the R0^(16/3) damping this shifts ATM energies by
  ~1e-7 relative, 1000× the tolerance. `scripts/export_reference_data.py` recovers the exact
  published values by rounding in Å (4 and 2 decimals) and converts with mctc-lib's factor.
- The fixture positions are unwrapped (fractional coordinates −0.6 … 1.8). The brute-force
  backend enlarges simple-dftd3's per-axis image count by the spread of the fractional
  coordinates, so its list is complete for any input; wrapping does not change the result.
- nvalchemiops: `max_neighbors` overflow is detected from `num_neighbors` and raises; shifts
  are consistent with unwrapped input; vectors are always recomputed in torch.
- Self-image neighbours (i, i, T ≠ 0) are included, (i, i, 0) excluded.
- `torch.compile` results agree with eager to ~1e-16 relative in float64, but not bit for bit:
  inductor reorders reductions.
