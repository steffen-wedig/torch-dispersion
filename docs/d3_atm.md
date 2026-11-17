# D3 ATM three-body term

`torch_dispersion.atm.atm_dispersion` evaluates the Axilrod–Teller–Muto (ATM) three-body term
of DFT-D3 with simple-dftd3 1.6.0's definition (`get_dispersion(s9 = 1) − get_dispersion(s9 =
0)`). It returns energy, forces and stress for molecules and fully periodic cells. It exists to
add the ATM term back to an MLIP trained on dispersion-free labels, next to a two-body D3 (the
GPU two-body codes, FourierD3 and nvalchemiops, omit ATM). The original specification is
archived in [`archive/design_torch_atm.md`](archive/design_torch_atm.md).

```python
from torch_dispersion.atm import atm_dispersion
from torch_dispersion.calculators.ase import TorchATMCalculator
from torch_dispersion.settings import ATMSettings

settings = ATMSettings(device="cuda")  # s9 = 1, alpha = 14, cutoffs 40 bohr
result = atm_dispersion(numbers, positions, cell, pbc=True, settings=settings)
result.energy, result.forces, result.stress  # eV, eV/Å, eV/Å³ (ASE sign convention)

atoms.calc = SumCalculator([mlip, two_body_d3, TorchATMCalculator(settings)])
```

To match FHI-aims numbers bit for bit, pass its unit constants:
`ATMSettings(bohr_in_angstrom=0.52917721, hartree_in_ev=27.211384500)`.

| setting | default | meaning |
|---|---|---|
| `s9`, `alpha`, `rs9` | 1.0, 14.0, 4/3 | ATM scale, the damping set's `alp` (the ATM uses `alpha + 2`), radius scale |
| `three_body_cutoff_bohr` | 40 | hard cutoff on all three legs |
| `three_body_cutoff_width_bohr` | 0 | > 0 multiplies each leg by a quintic switch on (cutoff − width, cutoff); use it for MD with a shortened cutoff |
| `coordination_cutoff_bohr` | 40 | hard CN cutoff |
| `neighbor_backend` | brute force | `BruteForceNeighbors()` or `NvalchemiNeighbors(max_neighbors=...)` |
| `centre_block_size`, `neighbor_tile_size` | 32, 512 | kernel shape: B centres × T × T neighbour pairs per call |
| `kernel_gradient` | analytic | `analytic` or `vjp` (`torch.func.vjp`, the reference) |
| `dtype`, `device`, `compile` | float64, cuda, True | |

## Definition (simple-dftd3 1.6.0)

Internally everything is in bohr and Hartree.

1. **CN:** CN_i = Σ_{j,T} 1 / (1 + exp(−16 (R^cov_ij / r_ij,T − 1))), with the D3 covalent
   radii (scaled by 4/3), over all j and lattice translations T with r ≤ 40 bohr (hard cutoff),
   including periodic images of i itself.
2. **C6:** Gaussian reference weights w_ia ∝ exp(−4 (CN_i − CN_ref(Z_i, a))²), normalised per
   atom, then C6_ij = Σ_ab w_ia w_jb C6_ref(Z_i a, Z_j b). An underflowed normalisation falls
   back to the reference with the largest CN_ref (weight 1, zero derivative).
3. **ATM:** for a triangle (i, j + T_j, k + T_k) with all three legs inside the cutoff,

       r0 = (rs9 R0_ij)(rs9 R0_ik)(rs9 R0_jk),   r1 = r_ij r_ik r_jk
       f_damp = 1 / (1 + 6 (r0 / r1)^((alpha + 2) / 3))
       e = s9 sqrt(|C6_ij C6_ik C6_jk|) f_damp (3 cos θ_i cos θ_j cos θ_k + 1) / r1³

   R0 is the zero-damping pairwise radius table (not the BJ radius). Triangles with any
   |C6| < eps are skipped.
4. **Counting:** E = (1/3) Σ_i Σ_{n < m ∈ N(i)} e(i, n, m) over the full neighbour list of
   each centre. Each triangle is visited from each corner, which also covers triangles
   through periodic images of one atom (the one-atom cell, where simple-dftd3 uses its 1/6
   factor).

## Implementation

1. **Graph stage** (autograd): positions and a zero strain → CN → weights → C6 matrix (N × N),
   compiled with dynamic shapes.
2. **Triple stage:** on the detached C6, a Python loop over blocks of B centres and pairs of
   T-wide tiles of their neighbour lists calls a fixed-shape compiled step (`atm/kernel.py`).
   It returns the tile's energy and its gradients with respect to the displacement vectors and
   C6, and adds them to running sums of dE/dR, the strain virial and dE/dC6. Memory is bounded
   by one tile, not by the ~10⁹ triangles of a liquid box.
3. **Chain closure:** one `torch.autograd.grad` through C6 adds the CN contribution.

The derivatives are hand-written: per-triangle math in one compiled pointwise graph, the row
and column sums and the one-hot scatters in a second. The kernel's only system-sized inputs (C6
and the running sums) are dynamic dimensions, so one compiled kernel serves every system,
cutoff and neighbour count, and MD steps do not recompile. See
[`implementation.md`](implementation.md) for the shared machinery, and the batch-wide variant
of this kernel that D4 uses.

## Validation

The oracle is `dftd3==1.6.0`, live and through the stored fixtures in `tests/fixtures/`
(FHI-aims unit constants). Tolerances: energy 1e-10 Hartree, gradient 1e-9 Hartree/bohr, virial
1e-9 Hartree, on molecules, small and skewed cells (including the one- and two-atom cells), the
fixture crystals and the six 238–256-atom liquid boxes (`slow`), on both neighbour backends,
eager and compiled, CPU and CUDA. Further tests cover the C6 model alone (through the two-body
kernel), finite-difference forces and stress, invariances, blocking, edge cases, float32 and
the ASE calculator.

## Performance

On one H200, a 250-atom liquid box at the full 40 bohr cutoff takes 0.33 s for energy, forces
and stress in float64 (0.09 s energy only), against 36 s for the single-threaded CPU oracle. At
25 bohr it takes 0.04 s, because the number of triangles grows roughly as the cutoff⁶. Going
from 40 to 25 bohr shifts the liquid box energies by −6 … −14 meV (1–2 %) and the forces by ≤
0.14 meV/Å. Memory stays below 0.8 GiB. Full tables: [`benchmarks/d3_atm.md`](benchmarks/d3_atm.md).

## Pitfalls

- The ATM exponent is `alpha + 2`; `rs9 = 4/3`; r0 and r1 are **products** over the three legs.
- R0 is the zero-damping pairwise radius table, not the BJ radius.
- C6 is the full CN-dependent D3 C6 with a 40 bohr hard CN cutoff and exp counting (k1 = 16). A
  different CN (such as FourierD3's `smooth_cut`) is a different term.
- **Reference-table precision.** tad-mctc's `vdw-pairwise.pt` carries ~1e-7 relative noise and
  tad-dftd3 builds `r4r2` in float32. Through the R0^(16/3) damping this shifts ATM energies by
  ~1e-7 relative, 1000× the tolerance. `scripts/export_reference_data.py` recovers the exact
  published values by rounding in Å (4 and 2 decimals) and converts with mctc-lib's factor.
- The fixture positions are unwrapped (fractional coordinates −0.6 … 1.8). The brute-force
  backend enlarges simple-dftd3's per-axis image count by the spread of the fractional
  coordinates, so its list is complete for any input.
- Self-image neighbours (i, i, T ≠ 0) are included, (i, i, 0) excluded.
- `torch.compile` results agree with eager to ~1e-16 relative in float64, but not bit for bit:
  inductor reorders reductions.
