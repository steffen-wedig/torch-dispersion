# Implementation

How the terms are computed, and what they share. The term-specific documents are
[`d4.md`](d4.md) (DFT-D4, including ORCA's D4rev) and [`d3_atm.md`](d3_atm.md) (the D3 ATM
term); the experimental cutoff-free ATM is in [`experimental/`](experimental/fourier_atm.md).

## Package layout

```
torch_dispersion/
  settings.py        pydantic settings: D4Settings, ATMSettings, the neighbour backends
  neighbors/         NeighborMatrix (compact padded full list); brute-force and nvalchemiops
                     backends; build_flat for a batch of systems
  core/              shared machinery: input checks and units (inputs), padding, batch layout
                     (batch), CN routine (coordination), C6 contraction (c6), tile layouts and
                     the tile loop (tiles), graph stage and chain closure (chain)
  d3/                D3 reference tables, CN counting functions, reference weights
  d4/                D4 model (CN, weights), EEQ-2019 charges (eeq), orchestration (dispersion)
  atm/               triple kernels: per-system (kernel), batch-wide with per-pair gradients
                     (flat); the real-space D3 ATM term (realspace)
  pairwise/          batch-wide two-body kernel with rational (BJ) damping (rational)
  calculators/       ASE calculators (ase), TorchSim model (torchsim, needs the extra)
  experimental/      methods whose API may change: fourier_atm (cutoff-free ATM)
```

The top-level package exports nothing but `__version__`; import from the submodules.

## Three stages

Every term is evaluated in the same three stages (`core/chain.py`):

1. **Graph stage, with autograd.** Positions and one strain leaf per system → CN (and, for D4,
   EEQ charges) → C6. This is O(N · neighbours) plus, for D4, the O(n²) EEQ solve per system.
   It is compiled as one function with dynamic shapes, so its backward is one compiled graph
   too.
2. **Tile stage, on detached C6.** The expensive pair and triple sums run as a Python loop over
   fixed-shape compiled kernel calls (`core/tiles.py`). Each call returns its energy and its
   explicit gradients with respect to the positions, the strain and C6, which are added to
   running sums. No autograd graph of the ~10⁹ triangles of a liquid box is ever built, so
   memory is bounded by one tile.
3. **Chain closure.** One `torch.autograd.grad` of the graph-stage outputs, with the
   accumulated dE/dC6 as `grad_outputs`, adds the implicit CN and charge contributions.

Internally everything is in bohr and Hartree; the API takes Å and returns eV, eV/Å and eV/Å³
(ASE sign conventions), with the conversion constants as settings.

## Neighbour lists

`NeighborMatrix` is a full list (both i → j and j → i) padded to a common width: int32 indices,
int8/int16 lattice shifts, a mask and per-row counts. It includes periodic self-images (i, i,
T ≠ 0) and excludes (i, i, 0). Displacement vectors are never taken from the backend; they are
recomputed in torch from the caller's (possibly strained) positions and cell, which is where
autograd enters.

- **Backends:** brute force (all pairs × all translations, the test reference; one system at a
  time) and nvalchemiops (warp; one `system_idx`-aware call for a whole batch, overflow of
  `max_neighbors` is an error). Use nvalchemiops for batches: the brute-force backend adds tens
  of milliseconds per call on batches of many small molecules.
- **One list per call** at the largest cutoff; the shorter ones (CN, EEQ CN, ATM) are derived
  with `restrict`, which compacts the rows.
- **Batches:** `build_flat` returns one matrix for S systems concatenated along the atom axis,
  with global indices; each row lists neighbours of its own system only.

## Batch layout

Atoms are flat (N rows, system by system, `core/batch.py`). Quantities that couple all atoms
of a system densely (the EEQ Coulomb matrix, C6) are per-system blocks (S, n_max, n_max) padded
to the largest system. Kernels read C6_ij from the flattened blocks at (system, local i, local
j). A single system is a batch of one, so `d4_dispersion` and the TorchSim model share one code
path.

## Kernels

| kernel | used by | layout | gradients |
|---|---|---|---|
| `pairwise/rational.py` | D4 two-body (and the D3 C6 test) | whole batch, tiles cut on use (`SliceLayout`) | analytic; forces from row sums (pair symmetry), dE/dC6 by scatter-add |
| `atm/flat.py` | D4 ATM | whole batch, `SliceLayout` with per-block tile widths | analytic, per pair: 3/2 × the centre-leg derivatives ([`notes/atm_pair_gradients.md`](notes/atm_pair_gradients.md)); forces from row sums |
| `atm/kernel.py` | D3 ATM, experimental FourierATM short-range part | one system, tiled copies with displacement vectors (`TileLayout`) | analytic (or `torch.func.vjp`); the far side's dE/dC6 by one-hot matmuls |

Common rules, which make each call a pure, fixed-shape function that compiles with
`fullgraph=True`:

- The heavy tensors are (B, T) and (B, T, T), with B centres per call and T neighbours per
  tile. Padded entries are neutralised with `torch.where` and safe values before any division,
  sqrt or power; there is no data-dependent control flow.
- Parameters are 0-d tensors, so one graph serves every parameter set and cutoff.
- The system-sized inputs (C6, the per-atom tables, the accumulators) are dynamic dimensions,
  so new systems and batches do not recompile.
- Sums accumulate in float64 even when the per-triangle math is float32.
- The triple kernels split the per-triangle math and its reductions into two compiled graphs
  (pointwise, then reduce): fused, inductor generated a register-starved persistent reduction
  ~4× slower on an H200.

**Tile widths of the batch-wide ATM kernel.** Each block of B centres gets the narrowest
power-of-two multiple of `min_neighbor_tile_size` (32) that holds its longest list, at most
`neighbor_tile_size` (512). A block of 15-atom molecules then runs (32, 32, 32) tiles instead of
(32, 512, 512). Full-width tiles run a graph with a static width; all narrower widths share one
graph in which the width is dynamic. A static graph per width exceeded dynamo's recompile
limit (8 per function, a hard error under `fullgraph`), and a single dynamic graph was 1.7×
slower on dense cells.

## Where the time goes

- **Without ATM** (s9 = 0) a D4 call is launch- and Python-bound for small systems and spends
  the rest in restricting the lists, the backward of the graph stage, the EEQ setup and the
  two-body tiles (see [`d4.md`](d4.md#torchsim-model-batched)).
- **With ATM** the triple tiles dominate on dense systems at the default 40 bohr: (B, T, T)
  float64 work per tile pair, and the number of triangles grows as about cutoff⁶. This is the
  place for a fused Triton kernel. The per-pair gradient form of `atm/flat.py` is what makes it
  practical: a program for (centre block, tile p, tile q) keeps its T × T triangles in
  registers, reduces rows and columns, and writes per-pair derivatives, with no N × N scatter.

## Tests and oracles

| term | oracle | tolerances (float64) |
|---|---|---|
| D4 (incl. D4rev parameter sets) | `dftd4==4.3.0` | energy 1e-10 Hartree, gradient 1e-9 Hartree/bohr, virial 1e-9 Hartree |
| D3 ATM | `dftd3==1.6.0` and stored fixtures | the same |
| FourierATM (experimental) | real-space sums at growing cutoffs | see [`experimental/fourier_atm.md`](experimental/fourier_atm.md) |

The kernels are also tested against each other (batch-wide against per-system ATM, analytic
against `torch.func.vjp`, compiled against eager), for recompilation, and against finite
differences.

```bash
uv run pytest -m "not slow and not gpu"   # fast CPU suite (~10 min on a compute node)
sbatch scripts/slurm/gpu_tests.sbatch     # GPU tests incl. slow, on one H200/B200 (DAIS)
sbatch scripts/slurm/flat_atm_verify.sbatch   # D4 benchmarks + CPU and GPU suites, one B200
```

`torch.compile` on the CPU needs a C++20 compiler for inductor; on DAIS run
`module load gcc/15` and `export CXX=$(which g++)` first.
