# ATM gradients on pairs: dropping the far side of each triangle

Status: verified numerically (2026-10-25). Implemented for D4 (2026-11-17) in the batch-wide
ATM kernel `atm/flat.py`, which `tests/test_atm_flat.py` compares with the per-system kernel
(energies to 1e-13, forces, virials and symmetrised dE/dC6 to 1e-11 relative). The D3 ATM
(`atm/kernel.py`, `atm/realspace.py`) and the Fourier short-range kernel still use the
per-system kernel with the far-side scatter. The note remains the reference for a Triton ATM
kernel.

## The problem it solves

The triple kernel visits, for each centre i, every unordered pair {n, m} of its neighbours, and
returns the energy and its derivatives with respect to the two centre legs (v_n, v_m) and to C6.
The derivative with respect to C6 of the far side (j, k) = (n, m) is the expensive part:

- it lands on an atom pair that is not in the centre's own neighbour list, so its slot is
  unknown;
- `_analytic_reduce` therefore scatters it with one-hot matmuls,
  `hot_n^T h_jk hot_m` with `hot_n`, `hot_m` of shape (B, T, Np);
- that costs O(B (Np T² + Np² T)) per tile, so the cost of a tile grows with the system size.
  This was the superlinear ATM gradient of the per-system D4 path (H200, job 712428): 0.84 s
  energy only but 11.7 s with forces at 2048 atoms.

The far side's position derivative (`g_jk`) is cheap in the torch kernel (batched matmuls), but
in a fused kernel it would also need a scatter onto (n, m).

## The identity

The ATM energy is a sum over triangles t = {a, b, c} of a function that is **symmetric** in the
three corners and depends only on pair quantities: the squared side lengths u_ab, u_bc, u_ca (the
angle factor follows from them by the law of cosines, the damping and the switching function are
per side) and the pair coefficients C6_ab, C6_bc, C6_ca.

The kernel computes E = (1/3) Σ_i Σ_{n<m ∈ N(i)} e(i, n, m): each triangle is visited three
times, once from each corner, with weight 1/3. In each visit, two of the three sides touch the
centre ("centre legs") and one does not ("far side"). Take any side ab of a triangle t:

- it is a centre leg in the visits from a and from b, and the far side in the visit from c;
- the derivative ∂e_t/∂x_ab (x = u or C6) is the same number in all three visits, because e_t
  is the same symmetric function seen from different corners.

The visits that see ab as a centre leg therefore contribute (2/3) ∂e_t/∂x_ab, and the full
derivative is

    ∂E/∂x_ab = (3/2) Σ_{visits with ab as a centre leg} (1/3) ∂e/∂x_ab.

So the far side never needs a derivative: scale the centre-leg derivatives by 3/2 and drop
`g_jk` and `h_jk` entirely. Every derivative then belongs to a pair (i, slot) of the centre's own
neighbour list, and the kernel's output is per pair, like the two-body kernel's:

- dE/du_(i, slot) and dE/dC6_(i, slot), no N × N output;
- forces and the virial follow from one pair scatter, dE/dR = Σ_pairs (dE/du) 2 v (to j, minus
  to i), virial Σ_pairs v ⊗ (dE/du) 2 v;
- the C6 chain takes a per-pair `grad_outputs`.

The energy itself is unchanged and still needs the far side: its length (from v_m − v_n, free)
and its C6_jk value (a read, not a scatter).

## Preconditions

The identity needs every triangle visited from all three corners with the same weight and the
same per-triangle value:

- full neighbour lists (i → j and j → i), as built today;
- the same cutoff and switching function on every side (the kernel's `valid` mask and
  `_switch` treat all three sides alike);
- a symmetric damping (r0 is the product over the three sides) and a symmetric C9
  (sqrt(C6 C6 C6), or the separable CP form of FourierATM's short-range kernel).

It would break with half lists, a centre-dependent cutoff, or a centre-only switch. Periodic
images are fine: the visits from the three corners are translated copies, and derivatives are
accumulated per atom pair.

## Verification

Patching the current kernel on CPU (float64, eager, 18 bohr cutoffs, `centre_block_size=8`,
`neighbor_tile_size=64`), compared with the unpatched kernel:

| system | max abs force (eV/Å) | dropping the far side's C6 derivative only | dropping it for C6 and positions |
|---|---|---|---|
| 20-atom cluster | 7.5e-4 | 9.7e-19 | 1.1e-18 |
| 3-atom cell | 8.1e-4 | 4.3e-19 | 3.3e-19 (virial 8.7e-19) |
| 5-atom cell | 2.0e-4 | – | 3.9e-19 (virial 5.2e-18) |
| 1-atom cell (self-image triangles) | 6.8e-22 | 5.5e-28 | – |

The energy is bit-identical (the patch touches only derivatives). The patch, for the record:

- `_analytic_pointwise`: return `1.5 * g_ij`, `1.5 * g_ik` and zeros for `g_jk`;
- `_analytic_reduce`: keep the energy and vector gradients, and build dE/dC6 from the centre
  legs only, `index_put_` (accumulate) of `1.5 * h_ij` at (centre, idx_n) and `1.5 * h_ik` at
  (centre, idx_m), where `h_ij = ½ e.sum(2) / C6_ij`, `h_ik = ½ e.sum(1) / C6_ik`.

## What it enables

- **Torch kernel:** removes the one-hot (B, T, Np) scatter, so a tile's cost no longer depends
  on Np. Without `g_jk` the reduce also loses its two (B, T, T) × (B, T, 3) matmuls. The
  remaining per-pair scatter is (B · T) entries per tile (`index_add`), or none at all if the
  accumulators become per-pair arrays. The `grad_pos` one-hot scatter in `_accumulate` can go
  the same way.
- **Triton kernel (later):** a program for (centre, tile p, tile q) keeps its T × T triangles in
  registers, reduces rows and columns, and writes per-pair derivatives with atomics only between
  the tile-pair programs of the same centre. With per-system C6 blocks for the C6_jk reads, the
  kernel runs over a work list spanning every system of a batch.
- **One scatter for both terms:** the D4 two-body and ATM kernels then produce the same per-pair
  outputs, so forces, virial and the C6 chain are shared.

Bit-identity with the current kernel is lost (different summation), at the 1e-18 level above.
Tests against the oracle are unaffected.
