# torch-dispersion

GPU-capable, `torch.compile`-able PyTorch implementations of dispersion corrections, with energy,
forces and stress for molecules and fully periodic cells. Each term reproduces a reference
code to its numerical precision, and runs on whole batches of systems for MD with MLIPs.

| term | entry point | reference | docs |
|---|---|---|---|
| DFT-D4: EEQ charges, two-body BJ + ATM, 118 functional parameter sets incl. ORCA's D4rev | `torch_dispersion.d4.d4_dispersion` | dftd4 4.3.0 | [`docs/d4.md`](docs/d4.md) |
| D3 Axilrod–Teller–Muto three-body term | `torch_dispersion.atm.atm_dispersion` | simple-dftd3 1.6.0 | [`docs/d3_atm.md`](docs/d3_atm.md) |
| *experimental:* cutoff-free D3 ATM (FourierATM) | `torch_dispersion.experimental.fourier_atm` | real-space sums | [`docs/experimental/`](docs/experimental/fourier_atm.md) |

Interfaces: the functions above (Å in; eV, eV/Å, eV/Å³ out, ASE conventions), ASE calculators
(`torch_dispersion.calculators.ase`) and a batched TorchSim model
(`torch_dispersion.calculators.torchsim.D4Model`). The top-level package exports nothing but
`__version__`; import from the submodules.

## Installation

```bash
uv sync                      # torch, ase, numpy, pydantic, nvalchemiops (warp neighbour lists)
uv sync --extra torchsim     # + torch-sim, for the TorchSim model
```

`torch.compile` on the **CPU** needs a C++20 compiler (inductor); on DAIS run
`module load gcc/15` and `export CXX=$(which g++)` first. CUDA compilation uses Triton and
needs nothing extra.

## Quick start

[`examples/quickstart.py`](examples/quickstart.py) computes D4 and the D3 ATM term for a molecule
and a periodic box, uses the ASE calculators, and runs a few MD steps:

```bash
uv run python examples/quickstart.py            # CPU
uv run python examples/quickstart.py --device cuda
```

### DFT-D4

```python
from torch_dispersion.calculators.ase import TorchD4Calculator
from torch_dispersion.d4 import d4_dispersion
from torch_dispersion.settings import D4Settings, NvalchemiNeighbors

# any of dftd4's d4.bj-eeq-atm entries; "wb97m-rev" is ORCA's wB97M-D4rev
settings = D4Settings.from_functional(
    "wb97m-rev", device="cuda", neighbor_backend=NvalchemiNeighbors()
)
result = d4_dispersion(numbers, positions, cell, pbc=True, settings=settings, charge=0.0)
result.energy, result.forces, result.stress, result.charges
atoms.calc = TorchD4Calculator(settings)
```

`atm=False` drops the three-body term (s9 = 0); other keyword arguments set any field
(cutoffs, kernel shapes, dtype). Use the nvalchemiops neighbour backend on the GPU: the default
brute-force backend is the test reference.

### Batched MD with TorchSim

```python
from torch_sim.models.interface import SumModel
from torch_dispersion.calculators.torchsim import D4Model

model = SumModel(mlip, D4Model(settings))  # energy (S,), forces (N, 3), stress (S, 3, 3)
```

Every stage runs on the whole batch at once: one neighbour list, one compiled graph stage (CN,
a batched EEQ solve with one total charge per system, C6), one two-body and one ATM tile loop
over all atoms, and one backward pass. Each system's result equals `d4_dispersion` on it to
rounding. A batch must be all periodic or all molecular.

### D3 ATM

```python
from torch_dispersion.atm import atm_dispersion
from torch_dispersion.settings import ATMSettings

result = atm_dispersion(numbers, positions, cell, pbc=True, settings=ATMSettings(device="cuda"))
```

## Performance

On one GPU, float64, compiled, energy, forces and stress, default cutoffs (two-body 60, ATM
40, CN 30 bohr). Molecular liquid boxes, H200 (job 913814):

| system | D4 with ATM | D4 without ATM | dftd4, 12 CPU cores |
|---|---|---|---|
| liquid box, 238–256 atoms | 0.16–0.30 s | 0.02–0.03 s | 2.9–6.5 s |
| liquid box, 1024 atoms | 0.78 s | 0.06 s | – |
| liquid box, 2048 atoms | 3.4 s | 0.14 s | – |

- **The ATM term dominates.** At 40 bohr each atom of a liquid has thousands of neighbours, and
  the triangles grow as about cutoff⁶; a shorter ATM cutoff with a switching width, or a fused
  Triton kernel, is where the remaining time is.
- **Molecular batches:** 64 molecules of 10–19 atoms with wB97M-D4rev take 23 ms per call
  (B200), on par with dftd4 on 12 CPU cores; 6 clusters of ~250 atoms take 33 ms against 0.49 s.
- **Batches of cells:** without ATM, a 15-cell batch of 5460 atoms takes 0.17 s (B200); with
  ATM, 12 molecular crystals of 6090 atoms take 4.9 s.

Details: [`docs/d4.md`](docs/d4.md), [`docs/benchmarks/`](docs/benchmarks/d4.md).

## Documentation

| document | contents |
|---|---|
| [`docs/d4.md`](docs/d4.md) | D4: definition, functionals and D4rev, TorchSim model, validation, timings |
| [`docs/d3_atm.md`](docs/d3_atm.md) | D3 ATM: definition, settings, validation, pitfalls |
| [`docs/implementation.md`](docs/implementation.md) | shared machinery: stages, neighbour lists, batch layout, kernels, compile policy |
| [`docs/benchmarks/`](docs/benchmarks/d4.md) | benchmark tables (D4, D3 ATM) |
| [`docs/notes/atm_pair_gradients.md`](docs/notes/atm_pair_gradients.md) | the per-pair ATM gradient identity behind the batch-wide kernel |
| [`docs/experimental/`](docs/experimental/fourier_atm.md) | FourierATM: method, accuracy, benchmark |
| [`docs/archive/`](docs/archive/design_torch_atm.md) | the original torch-atm specification |

## Tests

```bash
uv run pytest -m "not slow and not gpu"   # fast CPU suite (run it on a compute node)
sbatch scripts/slurm/gpu_tests.sbatch     # GPU tests incl. slow, on one H200/B200 (DAIS)
```

The oracles are `dftd4==4.3.0` and `dftd3==1.6.0` (dev dependencies), live and through the
stored fixtures in `tests/fixtures/`. The experimental tests are in `tests/experimental/`.

## Status

- D4 and the D3 ATM term are validated against their oracles; D4rev additionally against
  ORCA's published parameters (not yet against an ORCA run).
- Next: a fused Triton kernel for the ATM tiles, the remaining cost of D4 on dense systems.
