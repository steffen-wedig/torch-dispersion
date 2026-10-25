# torch-dispersion

GPU-capable, `torch.compile`-able PyTorch implementations of dispersion corrections, with energy,
forces and stress for molecules and fully periodic cells. Every term reproduces a reference code.

| term | entry point | settings | reference |
|---|---|---|---|
| D3 ATM three-body term, real space | `atm_dispersion` | `ATMSettings` | simple-dftd3 1.6.0 |
| D3 ATM, cutoff-free (FourierATM) | `fourier_atm` | `FourierATMSettings` | real-space sum at 80 bohr |
| DFT-D4, two-body + ATM, EEQ charges | `d4_dispersion` | `D4Settings` | dftd4 4.3.0 |

All of them take Å and return eV, eV/Å and eV/Å³ (ASE conventions). ASE calculators are
`TorchATMCalculator` (both ATM variants) and `TorchD4Calculator`. The real-space ATM was the
first term; its specification is [`docs/design.md`](docs/design.md).

## Installation

```bash
uv sync    # torch, ase, numpy, pydantic, nvalchemiops (warp neighbour lists)
```

`torch.compile` on **CPU** needs a C++20 compiler (inductor); on DAIS run `module load gcc/15`
and `export CXX=$(which g++)` first. CUDA compilation uses Triton and needs nothing extra.

## Quick start

[`examples/quickstart.py`](examples/quickstart.py) computes D4 and the D3 ATM term for a molecule
and a periodic crystal, uses the ASE calculators, and runs a few MD steps:

```bash
uv run python examples/quickstart.py            # CPU
uv run python examples/quickstart.py --device cuda
```

## D3 ATM (real space)

```python
import torch
from torch_dispersion import ATMSettings, atm_dispersion

settings = ATMSettings(device="cuda")  # s9 = 1, alpha = 14, cutoffs 40 bohr
result = atm_dispersion(numbers, positions, cell, pbc=True, settings=settings)
result.energy, result.forces, result.stress  # eV, eV/Å, eV/Å³ (ASE sign convention)
```

With ASE, e.g. summed with an MLIP and a two-body D3:

```python
from ase.calculators.mixing import SumCalculator
from torch_dispersion import TorchATMCalculator

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
| `centre_block_size`, `neighbor_tile_size` | 32, 512 | kernel shape: B centres × T × T neighbour pairs per call |
| `kernel_gradient` | analytic | `analytic` or `vjp` (reference) |
| `dtype`, `device`, `compile` | float64, cuda, True | |

## FourierATM (cutoff-free)

`fourier_atm` evaluates the same D3 ATM term with no three-body cutoff:

- **Undamped part:** a dense triple-dipole trace, −(s9/18) Σ_r μ_r ⟨Tr[(D_r A(q))³]⟩_q. It uses Bloch–Ewald dipole matrices A(q) on a small q-grid and a separable (CP) fit of D3's C9.
- **Short-range correction:** real-space triangles with all legs within 20 bohr get D3's exact damping and exact C9.

```python
from torch_dispersion import FourierATMSettings, fourier_atm, TorchATMCalculator

settings = FourierATMSettings(device="cuda")  # smooth_cut CN (as FourierD3); or coordination="d3"
result = fourier_atm(numbers, positions, cell, pbc=True, settings=settings)
atoms.calc = TorchATMCalculator(settings)  # same calculator, dispatches on the settings
```

On an H200, energy + forces + stress take 0.05 s for the 250-atom liquids and 1.9 s at 2048 atoms. The real-space term at 40 bohr takes 0.31 s and 11 s. Details: [`docs/benchmark_fourier.md`](docs/benchmark_fourier.md).

FourierATM is within 0.001–0.006 meV/atom of the real-space sum at 80 bohr. The 40-bohr sum (which the FHI-aims labels use) is 0.007–0.014 below it.

Use float64. Method, validation and limits: [`docs/fourier_atm.md`](docs/fourier_atm.md).

## DFT-D4

`d4_dispersion` evaluates the full DFT-D4 dispersion term (dftd4 4.3.0 `d4` model). It covers EEQ
charges, charge-scaled C6, the two-body rational damping and the ATM term. Energy, forces and
stress are supported for molecules and 3D cells.

```python
from torch_dispersion import D4Settings, TorchD4Calculator, d4_dispersion

settings = D4Settings.from_functional("r2scan", device="cuda")  # r2SCAN-D4 (bj-eeq-atm)
result = d4_dispersion(numbers, positions, cell, pbc=True, settings=settings, charge=0.0)
atoms.calc = TorchD4Calculator(settings)
```

`from_functional` looks up the damping parameters (s6, s8, s9, a1, a2, alp) in a frozen copy of
dftd4 4.3.0's `d4.bj-eeq-atm` table (118 functionals, `D4Settings.functionals()`). Names are
case-insensitive and otherwise exact (`r2scan-3c`, `pbe0_2`). `atm=False` sets s9 = 0, which
drops the ATM term and its 40 bohr neighbour list. Other keyword arguments set any field
(device, compile, cutoffs, neighbour backend); the parameters can also be given by hand,
`D4Settings(s8=..., a1=..., a2=...)`.

It reuses the tiled ATM kernel and adds a tiled two-body kernel. The periodic EEQ replicates
multicharge's fixed-range Ewald sums. Energies and virials match dftd4 to ~1e-17 Hartree, and
molecular gradients to ~1e-18 Hartree/bohr.

The oracle is `dftd4==4.3.0`. dftd4 up to 4.1.0 had a wrong periodic ATM gradient, fixed in
4.1.1; see [`docs/d4.md`](docs/d4.md).

## How it works (real-space ATM)

1. **Graph stage** (autograd): positions and a zero strain → CN → Gaussian reference weights
   → C6 matrix (N × N).
2. **Triple stage** (blocked): C6 is detached; a Python loop over blocks of B centres and pairs
   of T-wide tiles of their neighbour lists calls a fixed-shape compiled step that evaluates the
   tile's energy and its gradients with respect to the displacement vectors and to C6, and adds
   them to running sums of dE/dR, the strain virial and dE/dC6. Memory is bounded by one tile,
   not by the ~10⁹ triangles of a liquid box. The derivatives are hand-written
   (`kernel_gradient="analytic"`): per-triangle math in one pointwise graph, row/column sums,
   batched matmuls and one-hot matmuls (no atomics) in a second. `kernel_gradient="vjp"` uses
   `torch.func.vjp` instead; it is the reference (they agree to 1e-12) and ~4× slower.
3. **Chain closure**: `torch.autograd.grad(C6, ..., grad_outputs=dE/dC6)` adds the CN part.

Every unordered neighbour pair {n, m} of every centre i is visited once and the sum is divided
by 3, which covers triangles through periodic images of the same atom (the one-atom cell case).

**Recompilation control.** The kernel's heavy tensors are (B, T) and (B, T, T), fixed by the
settings; its only system-sized inputs (the C6 matrix and the running sums) have dynamic
dimensions and its parameters are 0-d tensors. So one compiled kernel serves every system,
cutoff and neighbour count; only a new
`(centre_block_size, neighbor_tile_size)`, dtype or device compiles a new one. Stage 1 (CN, C6)
is compiled with dynamic shapes. MD steps do not recompile (tested).

## Performance (real-space ATM)

On one H200, a 250-atom liquid box at the full 40 bohr cutoff takes 0.33 s for energy, forces and
stress in float64 (0.09 s energy only), against 36 s for the CPU oracle; at 25 bohr it is
0.04 s. Memory stays below 0.8 GiB. Details: [`docs/benchmark.md`](docs/benchmark.md), produced
by `sbatch scripts/slurm/benchmark.sbatch --oracle` and `scripts/benchmark_report.py`.

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
