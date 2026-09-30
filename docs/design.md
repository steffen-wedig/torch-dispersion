# torch-atm — design document

A standalone, GPU-capable, `torch.compile`-able PyTorch implementation of the D3
**Axilrod–Teller–Muto (ATM) three-body dispersion term**. It must reproduce
simple-dftd3 exactly (energy, forces, stress) for molecules and fully periodic cells.

Status: specification for hand-off (2026-09-30). The implementer owns a **new
repository**; nothing in this document asks for changes to `cluster_bulk_generalization`.

---

## 1. Why this exists

The DFT labels of the `cluster_bulk_generalization` project (r2SCAN, FHI-aims) include
simple-dftd3's D3(BJ) correction **with the ATM term switched on** (`s9 = 1`). The MLIPs
are now trained on dispersion-free labels, so dispersion has to be added back at
inference. The GPU two-body options that exist all omit ATM:

| package | two-body D3(BJ) | ATM | PBC | GPU |
|---|---|---|---|---|
| simple-dftd3 (`dftd3` on PyPI, the oracle) | yes | **yes** | yes | no (CPU, the wheel has no OpenMP) |
| FourierD3 (reciprocal space) | yes | no | yes | yes |
| nvalchemi-toolkit-ops 0.4.1 (`nvalchemiops`) | yes (+ Fourier variant) | **no** — every entry point documents "two-body only: three-body Axilrod-Teller-Muto (C9) terms not included" | yes | yes (warp) |
| tad-dftd3 | yes | yes (dense, molecular only) | **no** | yes (torch) |

Without ATM, a model plus two-body D3 runs on a different potential energy surface from
the one the training and benchmark structures were sampled on. Measured on all 6598
labelled liquid and crystal frames, dropping ATM makes the dispersion pressure 700–1260 bar
more negative (NPT densities about 4–5 % higher at 1 bar) and shifts the energy by about
−2.1 … −3.8 meV/atom. torch-atm closes that gap.

Two uses, in order of priority:

1. **Inference term** summed with the MLIP and a two-body D3 (ASE `SumCalculator` or an
   MD engine): energy, forces and stress in every step.
2. **Batch evaluation** of ATM energies on stored trajectories (reweighting studies):
   energy only, many frames.

## 2. Scope

In scope:
- The ATM energy, forces (−∂E/∂R) and virial/stress (∂E/∂ε) for molecules (no PBC) and
  fully periodic 3D cells, in float64 (reference) and float32.
- Everything ATM depends on: the D3 coordination number (CN), the Gaussian reference
  weighting, and the CN-dependent pairwise C6 coefficients. These are simple-dftd3's
  definitions, **not** FourierD3's (FourierD3's `smooth_cut` CN at 6 Å and its low-rank C6
  would make it a different term).
- Two neighbour-list backends behind one interface: a naive brute-force torch backend and
  an `nvalchemiops` backend.
- `torch.compile` support for the compute path (§6).
- An ASE calculator wrapper.
- A test-only two-body D3(BJ) energy, used to validate the C6 model on its own (§8, T1).

Out of scope:
- Production two-body dispersion (use FourierD3 or nvalchemiops).
- Hessians, 1D/2D periodicity (raise `ValueError` on mixed `pbc`), ghost atoms, per-atom
  energy partitioning beyond an optional `energies` output.
- Damping variants other than the ATM zero-damping, which is the same for every D3
  damping scheme in simple-dftd3.

## 3. Reference definition (simple-dftd3 v1.6.0)

The oracle is the **pinned** PyPI package `dftd3==1.6.0` (it reproduced the FHI-aims labels
to 1e-9 eV; FHI-aims embeds s-dftd3 1.1.0, whose ATM code gives identical results with
default cutoffs). Source files to read, at tag `v1.6.0` of
<https://github.com/dftd3/simple-dftd3>:

| what | file |
|---|---|
| ATM energy / gradient / virial | `src/dftd3/damping/atm.f90` (`get_atm_dispersion_energy`, `get_atm_dispersion_derivs`, `triple_scale`) |
| how ATM is called (`rs9`, `alp + 2`) | `src/dftd3/damping/rational.f90` (`get_dispersion3`) |
| cutoffs, smooth switching, lattice points | `src/dftd3/cutoff.f90` |
| CN | `src/dftd3/ncoord.f90` (and mctc-lib's `ncoord` for the unpartitioned path) |
| reference weighting, C6 | `src/dftd3/model.f90` (`weight_references`, `get_atomic_c6`, `weight_cn`) |
| orchestration (which cutoff feeds what) | `src/dftd3/disp.f90` (`get_dispersion_atomic_v2`) |

simple-dftd3 is LGPL-3.0. Re-implement from the equations; do not transcribe Fortran.
tad-dftd3 and tad-mctc (Apache-2.0) are the recommended source of the reference tables
(§5.2).

All internal quantities are in **atomic units** (bohr, Hartree). The equations below use
r in bohr.

### 3.1 Coordination number

    CN_i = Σ_{j, T} 1 / (1 + exp(−k1 (R_ij^cov / r_ij,T − 1))),   k1 = 16

- R_ij^cov = R_i^cov + R_j^cov, with the D3 covalent radii (mctc-lib `get_covalent_rad`,
  i.e. tad-mctc `radii.COV_D3`, already scaled by 4/3, in bohr).
- The sum runs over all atoms j and lattice translations T with r² ≤ cn_cutoff² (default
  **40 bohr**) and r² ≥ 1e-12. That includes periodic images of atom i itself (j = i, T ≠ 0).
- The cutoff is **hard**: no switching function on the CN.

### 3.2 Reference weights and C6

For atom i of element Z with references a = 1..N_ref(Z) (reference CNs CN_ref(Z, a)):

    w_i,a = exp(−wf (CN_i − CN_ref(Z,a))²) / Σ_b exp(−wf (CN_i − CN_ref(Z,b))²),   wf = 4

"Exceptional" fallback (NaN/Inf from an underflowed normalisation): the weight becomes
1 for the reference with the largest CN_ref of that element and 0 for all others. The
derivative of such a weight is 0.

    C6_ij = Σ_a Σ_b w_i,a w_j,b C6_ref(Z_i, a, Z_j, b)

### 3.3 ATM energy

Parameters: `s9` (1.0 for the labels), `alpha` (the damping parameter set's `alp`, 14.0),
`rs9 = 4/3` (a constant in `rational.f90`). Two easy-to-miss conventions:

- **The ATM exponent is `alp + 2`** (= 16): `get_dispersion3` passes `self%alp + 2`.
- **r0 and r1 are products of the three legs, not means.**

For a triangle of sites (i, j + T_j, k + T_k) with squared legs r²_ij, r²_ik, r²_jk:

    r0     = (rs9 R0_ij) (rs9 R0_ik) (rs9 R0_jk)         R0 = D3 pairwise vdW cutoff radii table (bohr)
    r2     = r²_ij r²_ik r²_jk,  r1 = sqrt(r2),  r3 = r2 r1,  r5 = r3 r2
    f_damp = 1 / (1 + 6 (r0 / r1)^((alp + 2)/3))
    ang    = 3/8 (r²_ij + r²_jk − r²_ik)(r²_ij − r²_jk + r²_ik)(−r²_ij + r²_jk + r²_ik) / r5 + 1/r3
           = (3 cos θ_i cos θ_j cos θ_k + 1) / (r_ij r_ik r_jk)^3
    e_ijk  = s9 · sqrt(|C6_ij C6_ik C6_jk|) · f_damp · ang

- A triangle contributes only if **all three** squared legs satisfy ε < r² ≤ cutoff3²
  (default cutoff3 = **40 bohr**; ε = machine epsilon on r²). Triangles where any C6 is
  below machine epsilon are skipped.
- The default switching width is 0 (hard cutoff). With `width3 > 0` (v1.6 option) each leg
  is multiplied by the quintic switch s(r) = x³(10 − 15x + 6x²), x = (cutoff − r)/width,
  on (cutoff − width, cutoff) (see `smooth_cutoff_r2`). Support this as an option; it is
  the right choice for MD with a shortened cutoff.
- R0 is the D3 **zero-damping pairwise cutoff-radius table** (mctc-lib/tad-mctc
  `VDW_PAIRWISE`), not the BJ radii a1·sqrt(3 Q) + a2.

**Counting.** simple-dftd3 loops i ≥ j ≥ k over home-cell atoms, j and k over all
translations, with `triple_scale` = 1 (all distinct), 1/2 (exactly two indices equal),
1/6 (all equal). An equivalent formulation that suits a neighbour list:

    E = (1/3) Σ_{i in cell} Σ_{unordered pairs {n, m} ⊂ N(i), n ≠ m} e(i, n, m)

where N(i) is the set of sites (j, T) with 0 < r_ij,T ≤ cutoff3, and both n and m in N(i)
automatically satisfy the i-legs; only the n–m leg must still be masked (r_nm ≤ cutoff3,
r_nm² > ε). Each translation class of triangles appears once per vertex (3×), which gives
the 1/3. This covers triangles through images of the same atom (for example i, i + T, j),
which matter in small cells. It is *derived*, not copied from the oracle, so tests T3/T4
(§8) must confirm it, including a one-atom cell where every triangle is a self-image triangle.

### 3.4 Derivatives and sign conventions

- Forces F = −∂E/∂R. The oracle returns `gradient` = +∂E/∂R (Hartree/bohr).
- The oracle's `virial` is the strain derivative ∂E/∂ε (extensive, Hartree). The ASE stress
  is `0.5 (virial + virialᵀ) / V` (eV/Å³ after conversion), with positive stress = tension.
  Pressure P = −tr(σ)/3.
- C6 depends on positions through CN, so the gradient has an explicit part (∂e/∂r at fixed
  C6) and a CN part (∂E/∂C6 · ∂C6/∂CN · ∂CN/∂R). With autograd this comes for free
  provided CN and C6 stay in the graph. It is the most common source of force errors
  (tested separately, §8 T5).

## 4. Units

The public API takes **Å and eV** and returns eV, eV/Å, eV/Å³. Conversion constants are
settings with defaults `bohr_in_angstrom = ase.units.Bohr` and `hartree_in_ev =
ase.units.Hartree`. Callers that must match FHI-aims' numbers bit-for-bit pass
`bohr_in_angstrom = 0.52917721`, `hartree_in_ev = 27.211384500` (the fixtures in §9 were
generated with these). Tests always use the fixture manifest's constants.

## 5. Architecture

```
torch_atm/
  settings.py          pydantic models (ATMSettings, NeighborBackend union, Units)
  data.py              reference tables → tensors (per device/dtype cache)
  neighbors/
    base.py            NeighborMatrix dataclass + NeighborBackend protocol
    brute_force.py     naive torch backend (all pairs × all translations)
    nvalchemi.py       nvalchemiops backend
  coordination.py      CN from a NeighborMatrix
  c6.py                reference weights, C6 matrix
  kernel.py            compiled triple-block kernel (§6)
  atm.py               orchestration: atm_dispersion(...) -> ATMResult
  two_body.py          TEST-ONLY D3(BJ) two-body energy (validates C6, §8 T1)
  ase_calculator.py    TorchATMCalculator
tests/
```

### 5.1 Configuration (pydantic, yaml-loadable, discriminated on `kind`)

```python
class BruteForceNeighbors(BaseModel):
    kind: Literal["brute_force"] = "brute_force"
    pair_block_size: int = 4096  # bounds memory of the N x N x n_translations scan


class NvalchemiNeighbors(BaseModel):
    kind: Literal["nvalchemi"] = "nvalchemi"
    method: Literal["naive", "cell_list"] | None = None  # None: library auto-select
    max_neighbors: int | None = None  # None: estimate + headroom, overflow is an error


NeighborBackend = Annotated[BruteForceNeighbors | NvalchemiNeighbors, Field(discriminator="kind")]


class ATMSettings(BaseModel):
    s9: float = 1.0
    alpha: float = 14.0  # the damping set's alp; the ATM uses alp + 2
    rs9: float = 4.0 / 3.0
    three_body_cutoff_bohr: float = 40.0
    three_body_cutoff_width_bohr: float = 0.0
    coordination_cutoff_bohr: float = 40.0
    centre_block_size: int = 16  # centres per compiled kernel call (§6)
    neighbor_backend: NeighborBackend = BruteForceNeighbors()
    bohr_in_angstrom: float = ase.units.Bohr
    hartree_in_ev: float = ase.units.Hartree
    dtype: Literal["float64", "float32"] = "float64"
    device: str = "cuda"
    compile: bool = True
```

`s9 = 0` must return zeros without doing work. Validate eagerly: cutoffs > 0,
0 ≤ width < cutoff, and `pbc` either all True or all False.

### 5.2 Reference data

Needed per element: D3 covalent radius (bohr), reference CNs, number of references; per
element pair: the reference C6 block (up to 7 × 7) and the vdW cutoff radius R0.
Recommended source: `tad-dftd3` (`reference-c6.pt`, `reference.py` for reference CNs) and
`tad-mctc` (`data.radii.COV_D3`, `data.radii.VDW_PAIRWISE`). Either depend on them at
runtime, or export them once into a versioned `.pt` in the package with a checksum test.
Do not trust the tables until T1 passes: T1 compares the full CN → weights → C6 chain
against the oracle's two-body energy.

### 5.3 Neighbour lists

One output type, used by CN and ATM alike:

```python
@dataclass
class NeighborMatrix:
    indices: Tensor  # (N, M) int64, padded with N (a dummy index)
    shifts: Tensor  # (N, M, 3) integer lattice shifts, 0 in padding
    mask: Tensor  # (N, M) bool, True for real neighbours
    count: Tensor  # (N,) neighbours per atom
```

- **Full list** (both i→j and j→i), padded to a common width M. It excludes (i, i, T = 0)
  and **includes** (i, i, T ≠ 0) self-images.
- Indices and shifts are non-differentiable. Displacement vectors are always recomputed
  in torch from the *caller's* positions and the (possibly strained) cell:
  `r_ij = pos[j] − pos[i] + shift @ cell`. Autograd for forces and stress flows through
  this, never through the backend.
- Two lists per evaluation, one per cutoff (CN at 40 bohr, ATM at cutoff3). When
  cutoff3 ≤ cn_cutoff, build one list at the larger cutoff and derive the other by masking.
- **Brute force** (reference, CPU/GPU): enumerate translations as in `get_lattice_points`
  (all integer combinations within the per-axis repeat count for the cutoff), compute all
  N × N × n_T distances in blocks of `pair_block_size`, and keep r² ≤ cutoff² with r² > ε.
  O(N² n_T) is fine for ≤ 1000 atoms. Correctness matters more than speed here: it is the
  test oracle for the other backend.
- **nvalchemiops** (`nvalchemiops.torch.neighbors.neighbor_list`, package
  `nvalchemi-toolkit-ops[torch]`, warp-based, Apache-2.0): call with `cell`, `pbc`,
  `cutoff` in the caller's length unit, matrix format (the default, returns
  `(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)` with PBC, int32, padded with
  `fill_value`), convert to `NeighborMatrix`. Traps:
  - At the 40 bohr (21 Å) default, a liquid at 0.1 atoms/Å³ has **~4400 neighbours per
    atom**. Set `max_neighbors` explicitly, and treat `num_neighbors > max_neighbors` as a
    hard error (never silently truncate).
  - `wrap_positions=True` wraps positions internally. Check that the returned shifts are
    consistent with the *unwrapped* input positions: recompute every r_ij in torch and
    assert r² ≤ cutoff² (+ tolerance) and that the neighbour set equals brute force (T7).
  - Cells smaller than the cutoff (the crystals here: 4–8 Å cells vs a 21 Å cutoff) need
    many images. Confirm the backend returns multiple shifts for the same (i, j), including
    self-images.

### 5.4 Memory and the gradient strategy

At the default cutoff a 250-atom box has ~250 × 4400²/2 ≈ 2.4e9 (centre, pair) candidates,
roughly half of them inside the n–m cutoff. The autograd graph of that cannot be kept, so
the computation is split in two stages:

1. **Graph stage (small):** positions (and a zero strain tensor `ε`, requires_grad, for
   stress) → strained cell and positions → CN → weights → C6 matrix (N × N). Keep this
   graph.
2. **Triple stage (large, blocked):** detach C6 into a leaf `c6_leaf` (requires_grad).
   Loop over blocks of `centre_block_size` centres. For each block call the compiled kernel
   (§6), which returns the block energy **and** its gradients with respect to the block's
   displacement vectors and to `c6_leaf`. Scatter-accumulate:
   - dE/dr_in into forces on i and n (and the strain virial Σ r ⊗ dE/dr),
   - dE/dC6 into a dense N × N accumulator.
3. **Close the chain:** `torch.autograd.backward(C6, grad_tensors=dE_dC6)` through the
   stage-1 graph adds the CN contribution to the position and strain gradients.

This is exact, keeps memory bounded by one block, and makes each kernel call a
fixed-shape pure function, which is what `torch.compile` needs. For energy-only calls
(reweighting), skip the gradient outputs entirely.

## 6. torch.compile requirement

The compute path must work under `torch.compile`, and the **triple kernel must compile
with `fullgraph=True`**. Requirements:

- **Static shapes in the kernel.** Inputs to `kernel.triple_block(...)` are fixed-shape
  padded tensors: `vectors (B, M, 3)`, `mask (B, M)`, `neighbor_index (B, M)`, the block's
  centre indices `(B,)`, `c6 (N, N)` (or pre-gathered `(B, M)` and `(B, M, M)` C6 slices),
  `r0 (B, M, M)` (or gathered on the fly from a species-pair table), scalars as 0-d
  tensors. The last block is padded to B with masked-out dummy centres, never shortened.
- **No data-dependent Python control flow** inside the kernel: no `.item()`, no
  `nonzero`/boolean indexing, no early `return` on "no triples", no Python loops over
  atoms. Masking uses `torch.where` with safe denominators (replace masked r by 1 before
  dividing, as tad-dftd3 does).
- **Gradients inside the compiled region.** The kernel returns `(energy, dE_dvectors,
  dE_dc6)`. Implement it with `torch.func.grad_and_value` (or `torch.func.vjp`) over a
  pure energy function, or with hand-written analytic derivatives. Both must compile
  fullgraph and agree with each other to 1e-12 in float64 (T9).
- **The orchestration loop stays eager** (a Python loop over blocks, scatter-adds, the
  stage-3 backward). It may be compiled later; it is not required.
- **Neighbour-list construction is outside the compiled region.** Brute force may be
  compiled separately (fixed N and n_T). nvalchemiops calls are warp custom ops, so treat
  them as graph breaks. Do not require fullgraph across them.
- **Recompilation control.** N, M and n_T change between systems. Use bucketed padding:
  round M up to a multiple of 64 and N up to a multiple of `centre_block_size`, so an MD
  run with slowly varying neighbour counts reuses one graph. Alternatively mark N and M
  dynamic (`torch._dynamo.mark_dynamic`). Document which is used, and test that 50
  MD-like steps (small displacements, same bucket) trigger zero recompilations (T9).
- **CN and C6 functions** (stage 1) must also compile fullgraph on fixed-shape inputs.
- `compile=False` must give bit-identical float64 results (eager is the debugging path).

## 7. Public API

```python
@dataclass
class ATMResult:
    energy: Tensor  # () eV
    forces: Tensor | None  # (N, 3) eV/Å
    stress: Tensor | None  # (3, 3) eV/Å³, ASE sign convention, symmetrized; None without PBC
    virial: Tensor | None  # (3, 3) eV, dE/dε (extensive)


def atm_dispersion(
    numbers: Tensor,  # (N,) int
    positions: Tensor,  # (N, 3) Å
    cell: Tensor | None,  # (3, 3) Å, rows are lattice vectors
    pbc: bool,
    settings: ATMSettings,
    compute_forces: bool = True,
    compute_stress: bool = True,
) -> ATMResult: ...


class TorchATMCalculator(ase.calculators.calculator.Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, settings: ATMSettings, **kwargs): ...
```

Caching: reference tensors per (device, dtype). The calculator re-checks the species
sequence on every call and rebuilds its caches when it changes. (FourierD3's calculator
does not do this and silently returns wrong numbers; do not copy that.)

## 8. Test plan (pytest; the oracle is `dftd3==1.6.0`)

The ATM reference for any structure is `get_dispersion(s9 = 1) − get_dispersion(s9 = 0)`
from `dftd3.interface.DispersionModel` with `RationalDampingParam(s6, s8, s9, a1, a2,
alp)`, for energy, gradient and virial. Cutoffs are set with
`model.set_realspace_cutoff(disp2, disp3, cn, width2, width3)`. The oracle exposes neither
CN nor C6, hence T1. Damping parameters for all tests (the labels' r2SCAN set):
`s6 = 1.0, s8 = 0.78981345, a1 = 0.49484001, a2 = 5.73083694, alp = 14.0, s9 = 1.0`.

Tolerances (float64): energy 1e-10 Hartree absolute (or 1e-10 relative for large boxes),
gradient 1e-9 Hartree/bohr, virial 1e-9 Hartree. float32: document the achieved error;
target 1e-4 relative in energy.

Mark fast tests (< 1 s each, molecules and small cells, shortened cutoffs where noted) as
default, and the full-cutoff liquid boxes as `slow`. Every test parametrizes over
`neighbor_backend ∈ {brute_force, nvalchemi}` and `compile ∈ {False, True}` unless noted,
and over `device ∈ {cpu, cuda}` when CUDA is available.

- **T1 C6 model via the two-body energy.** The test-only `two_body.py` computes D3(BJ)
  E_2 = −½ Σ_{i,j,T} C6_ij [s6 / (r⁶ + R⁶) + s8 · rr_ij / (r⁸ + R⁸)], with
  rr_ij = 3 · r4r2_i · r4r2_j (so C8 = C6 · rr_ij), R = a1 · sqrt(rr_ij) + a2, the `r4r2` table
  (tad-dftd3 `data.r4r2` = simple-dftd3 `data/r4r2.f90`), a 60 bohr hard cutoff, and this
  package's CN and C6. Read `rational.f90` (`get_dispersion_energy`) to confirm the form
  before coding. Compare against the oracle at `s9 = 0` for all fixture molecules and
  crystals, plus a random cluster with every element present (H, C, N, O, F, S, Br). This
  isolates table and weighting errors from ATM errors.
- **T2 molecular ATM.** Energy, gradient and per-molecule sanity on the fixture monomers,
  water trimer, benzene dimer, and a random 60-atom cluster (with close contacts to
  exercise the damping).
- **T3 small periodic cells.** Fixture crystals (triclinic, 10–34 atoms, cells smaller than
  the cutoff); a one-atom fcc Ar cell (only self-image triangles: tests the 1/6 case); a
  two-atom cell (the 1/2 cases); a strongly skewed cell. Energy, gradient, virial.
- **T4 liquid boxes** (`slow`). The six fixture boxes (238–256 atoms) against the stored
  references at `default` and `disp3_25` cutoffs. Plus the provenance check: the
  FHI-aims-printed D3 minus the oracle two-body equals the fixture ATM (already verified to
  1e-9 eV at export).
- **T5 derivatives without the oracle.** Central finite differences of this package's own
  energy for forces (random atoms, h = 1e-4 Å) and for the virial (strain FD). A variant
  with CN held fixed (C6 detached) vs full isolates the CN-chain contribution.
- **T6 invariances.** Rigid translation and rotation (molecules), atom permutation (forces
  permute), wrapping positions across the cell (identical result), and supercell
  consistency: E(2×1×1) = 2 E(cell), with tiled forces and equal stress.
- **T7 backend equivalence.** For fixtures and random cells, the nvalchemiops neighbour
  set (sorted (i, j, shift) tuples) equals brute force exactly, at both cutoffs. Include
  unwrapped input positions and a cell smaller than the cutoff. `max_neighbors` overflow
  raises.
- **T8 cutoff controls.** `three_body_cutoff_bohr ∈ {20, 25, 40}`,
  `coordination_cutoff_bohr ∈ {25, 40}` and `three_body_cutoff_width_bohr > 0` against the
  oracle with the same `set_realspace_cutoff` arguments.
- **T9 compile.** (a) The kernel compiles with `fullgraph=True`; (b) compiled equals
  eager bit-for-bit in float64 on T2–T4 structures; (c) `torch.func` gradients equal the
  analytic gradients if both exist; (d) zero recompilations across 50 perturbed steps in
  the same bucket (count via `torch._dynamo.utils.counters` or `torch._logs`
  `recompiles`); (e) a bucket change recompiles exactly once.
- **T10 blocking.** Results independent of `centre_block_size ∈ {1, 7, 16, N}` and of
  `pair_block_size`.
- **T11 edge cases.** `s9 = 0` returns zeros; a single atom; two atoms (no triangles);
  coincident atoms raise; mixed `pbc` raises; empty neighbour rows; the exceptional-weight
  fallback of §3.2 (force it with a CN far beyond every reference, e.g. by feeding an
  artificial CN to the weighting function, and check the weights are finite, one-hot on
  the largest reference CN, with zero derivative).
- **T12 ASE calculator.** Energy, forces and stress through `atoms.get_*` equal the
  functional API. Changing species on a reused calculator gives the correct (rebuilt)
  answer. Also verify inside `SumCalculator` with a dummy calculator.

Performance (benchmark script, not a test): wall time and peak memory per liquid box on
one H200 at cutoff3 ∈ {25, 30, 40} bohr, float64 and float32, eager vs compiled,
energy-only vs energy + forces + stress; and the CPU oracle time for comparison. Report
triangles per second. This decides whether MD uses the full or a shortened cutoff.

## 9. Fixtures (provided)

`/dais/u/wedigs/cluster_bulk_generalization/docs/torch_atm/fixtures/` (read-only for the
implementer; copy the three files into the new repository's `tests/fixtures/`):

- `structures.extxyz` — 19 structures, Å, `info["name"]`: per liquid (benzamide,
  bromobenzene, cyclohexane, dmso, emim_ntf2, glycerol) one MD box (`<liquid>_liquid`,
  cubic, 238–256 atoms), one crystal (`<liquid>_crystal`, triclinic, 10–34 atoms) and each
  distinct molecule (`<liquid>_monomer_<formula>`, no cell; emim_ntf2 has cation and
  anion). Elements: H, C, N, O, F, S, Br.
- `references.npz` — keys `<name>/<cutoff set>/{atm_energy, atm_gradient, atm_virial,
  two_body_energy}`, oracle output in atomic units (Hartree, Hartree/bohr, Hartree).
  Cutoff sets: `default` (disp2 60, disp3 40, cn 40 bohr) and `disp3_25` (disp3 25).
- `manifest.json` — oracle version, damping parameters, unit constants (positions were
  converted with 1 bohr = 0.52917721 Å), structure list with provenance.

Magnitudes, for orientation: ATM = +0.53 … +0.90 eV per liquid box (about 2.2–3.6
meV/atom, repulsive), +0.02 … +0.09 eV per crystal cell, ≤ 2 meV per monomer. Cutting the
three-body cutoff from 40 to 25 bohr changes box energies by −6 … −14 meV (1–2 %) and
forces by ≤ 0.14 meV/Å.

The generator is `/dais/u/wedigs/cluster_bulk_generalization/scripts/export_torch_atm_fixtures.py` (CPU,
~1 min on 32 cores; the simple-dftd3 1.6.0 ATM with gradient takes about 10 s per box).

## 10. Milestones

1. Data tables + CN + weights + C6 + test-only two-body; T1 green (brute force, eager, CPU).
2. ATM energy, eager, brute force; T2–T4 energies, T6, T8.
3. Blocked gradient strategy (§5.4); forces and virial; T2–T5, T10.
4. nvalchemiops backend; T7, then the whole suite on both backends.
5. torch.compile; T9; the whole suite with `compile=True`.
6. ASE calculator; T11, T12; benchmark report; README with usage and the pitfalls below.

## 11. Pitfalls checklist

- ATM exponent is `alp + 2`, not `alp`; `rs9 = 4/3`; r0 and r1 are **products** over the
  three legs.
- R0 is the zero-damping pairwise radius table, not the BJ radius.
- C6 is the full CN-dependent D3 C6 with a **40 bohr hard CN cutoff** and exp counting
  (k1 = 16). A different CN is a different term.
- `sqrt(abs(C6_ij C6_ik C6_jk))`, and triangles with any C6 < ε are skipped.
- All three legs must be inside the cutoff; ε-exclusion applies per leg.
- Self-image neighbours (i, i, T ≠ 0) must be present; (i, i, 0) must not.
- Units: bohr/Hartree internally; conversions are settings; the fixtures use the aims
  constants.
- Virial sign and symmetrization (§3.4); stress = virial/V in the ASE convention.
- nvalchemiops: `max_neighbors` overflow, int32 shifts, internal wrapping; always
  recompute vectors in torch.
- Autograd memory: never build the full triple graph; use §5.4.
- float32: the ~1e9-term sum needs blockwise accumulation in float64 even when the
  per-triangle math is float32.
- Do not set process-global torch flags (FourierD3's constructor flips TF32 globally). If
  TF32 matters, scope it with a context manager.

## 12. Integration notes (for the consuming project, not the implementer)

- Inference dispersion = MLIP + two-body D3 + torch-atm. With FourierD3 as the two-body,
  the remaining mismatch to the labels is FourierD3's CN/C6 approximation: at most
  0.09 meV/atom, 40 bar and 0.17 meV/Å RMS in forces across all six liquids, measured by
  `scripts/analyze_fourier_d3_shift.py`. A two-body term using the §3.1–3.2 CN would remove
  that too. simple-dftd3 ≥ 1.6 also has its own Ewald two-body (`set_ewald_summation`,
  CPU), which is useful as a second reference.
- Choose cutoff3 from the benchmark. A shortened cutoff with a switching width keeps MD
  energy-conserving, and its energy offset versus the labels is nearly constant per box
  (1–2 % of ATM).
