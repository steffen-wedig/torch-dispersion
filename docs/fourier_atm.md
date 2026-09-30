# FourierATM

A cutoff-free version of the D3 ATM three-body term (`torch_atm.fourier`). It is built on the same CN-based D3 coefficients as the real-space term and involves no learned polarisabilities. It is meant to sit next to FourierD3 (two-body) and can share FourierD3's `smooth_cut` coordination number.

Status (2026-09-30): implemented and tested.

- **API:** `fourier_atm(...)` with `FourierATMSettings`, or `TorchATMCalculator(FourierATMSettings(...))` for ASE.
- **Throw-away probes:** `/dais/u/wedigs/fourier_atm_probes/`. The numbers in §2 come from them.

---

## 1. Method

### 1.1 Why not a mesh (SPME) method

FourierD3 works because the two-body energy is a density–density convolution, Σ_ij c_i c_j φ(r_ij). ATM is a cyclic three-body sum. Written with pair kernels it is a matrix trace, not a convolution, so there is no particle-mesh version of it.

### 1.2 Triple-dipole trace

With T(r) = (3 r̂r̂ᵀ − 1)/r³ = ∇∇(1/r):

```
Tr(T_ij T_jk T_ki) = −3 (1 + 3 cos θ_i cos θ_j cos θ_k) / (r_ij r_jk r_ki)^3 = −3 · ang
```

This is the third-order term of MBD.

### 1.3 Separable C9

```
C9_ijk ≈ Σ_r μ_r a_ir a_jr a_kr,    a_ir = Σ_p w_ip(CN_i) V_pr
```

- V and μ are a symmetric CP decomposition of the reference tensor C9ref_pqs = sqrt(C6ref_pq C6ref_qs C6ref_sp) over the references of the species present (`fourier/cp.py`). This is the three-body analogue of FourierD3's eigen-decomposition of C6ref.
- The fit works in normalised form, Y = C9ref / (d_p d_q d_s) with d = sqrt(diag C6ref). It uses LBFGS with restarts and takes about 1.5 s per species set. It is cached per process.
- This is the D3 definition of C9, approximated. It is not an independent physical C9. The same engine accepts other coefficients, for example Casimir–Polder node values a_ir = α_i(iω_r), μ_r = W_r.

### 1.4 Undamped energy of all triangles

```
E_bare = −(s9/18) Σ_r μ_r <Tr[(D_r A(q))^3]>_q,    D_r = diag(a_r) ⊗ 1₃
```

- **The matrix A(q):** the 3N × 3N Bloch dipole matrix A_ij(q) = Σ_n T(x_j − x_i + n) e^{iq·n}, without the (i = j, n = 0) term (`fourier/dipole.py`). It is evaluated by Ewald summation:
  - an erfc real-space part over neighbour-list pairs;
  - one (3N × n_k)(n_k × 3N) complex matmul for the reciprocal part;
  - a self term.
- **Molecules:** one real matrix. The result is exact, with no cutoff.
- **Why average over q:** the q-average keeps closed triangles. The only other survivors are polygons that wind around the supercell of the grid.
- **The q-grid:** shifted off Γ, q_f = (k + ½)/n, and reduced by time reversal. The default grid is the smallest one with n × (plane spacing) ≥ 70 bohr, which is 3³ for the 250-atom liquids.

### 1.5 The q → 0 singularity (not in the plan)

As q → 0, the G = 0 reciprocal term tends to β u uᵀ with β = −4π/V and u = q̂ ⊗ 1_N. It depends on the direction of q, so Tr[(DA)³] is discontinuous at Γ. The q-average then converges only as O(1/N_q), measured as err·n³ ≈ constant.

The jump is known exactly. With M = D A_r(0), where A_r(0) is the q = 0 matrix without that term, α = Σ a_i, and bsum the 3 × 3 sum over atom blocks:

```
F(q̂) = β³α³ + q̂ᵀKq̂,   K = 3β bsum(DADAD) + 3β²α bsum(DAD)
```

An auxiliary function F(unit(q+G)) e^{−|q+G|²/σ²} has the same singularity and a known integral. Subtracting its grid average and adding that integral (`singular_correction`) turns the O(1/N_q) error into a much smaller one.

On a monomer in a 30–34 bohr box, the error against n = 6 was:

| grid | 2³ | 3³ | 4³ |
|---|---|---|---|
| uncorrected (Ha) | 2.9e-6 | 1.0e-6 | 4.4e-7 |
| corrected (Ha) | −7.4e-7 | −8.6e-8 | −1.8e-8 |

### 1.6 Damping: short-range correction

D3's triangle damping fdmp = 1/(1 + 6 Π_l (r0_l/r_l)^{16/3}) does not factorise into pair factors. So:

```
E = E_bare + Σ_{legs ≤ r_s} [C9_D3 fdmp − C9_CP] ang s(r_ij) s(r_ik) s(r_jk)
```

- The correction is the real-space tile kernel with a second per-atom table a (`fourier/kernel.py`), with analytic and vjp gradients.
- Default r_s = 20 bohr with a 3-bohr quintic switch.
- Inside r_s the result is exact D3. Outside it is undamped D3 with the CP C9.
- The CP error cancels exactly inside r_s, because the same CP C9 appears in the trace and in the subtraction.

### 1.7 Gradients

- **Stage 1:** CN → weights → C6 and a, with autograd.
- **Stage 2:** the short-range tiles on detached C6 and a. They accumulate dE/dR, the virial, dE/dC6 and dE/da.
- **Stage 3:** the trace (float64) with autograd through a, the positions and the cell (the strain trick). One `autograd.grad` closes both chains.

## 2. Accuracy

**Molecules, exact limits.**

- With r_s larger than the molecule, the result equals the real-space D3 at 60 bohr to 1e-10 relative, and forces to 1e-12.
- With finite r_s, it matches a brute-force triangle sum of the same definition to 1e-11.

**Periodic, against brute force.** A monomer in a 16 × 17 × 18 Å box, compared with the real-space triangle sum of the same definition at 120 bohr. This comparison is absolute because the total is a 400× cancellation:

| q-grid | error (eV) |
|---|---|
| 2³ | 2.0e-5 |
| 3³ | 2.5e-6 |
| 4³ | 6e-7 |

**Liquids, against the real-space D3 at 80 bohr** (D3 CN, meV/atom). Reference totals are 2.46–3.60 meV/atom. The 80-bohr sum still lacks about +0.0013 meV/atom of tail:

| liquid | FourierATM − D3@80 | D3@40 − D3@80 |
|---|---|---|
| benzamide | −0.0022 | −0.0091 |
| bromobenzene | −0.0044 | −0.0137 |
| cyclohexane | −0.0007 | −0.0065 |

What the table shows:

- **Remaining error.** The remaining −0.001 to −0.006 meV/atom comes from triangles beyond r_s being undamped. At r_s = 15 bohr it would be up to 0.08 meV/atom.
- **Closer than the labels.** FourierATM is closer to the cutoff-converged D3 than the 40-bohr sum (and so the FHI-aims labels) is.

**Derivatives.** They match central differences to 1e-10 with the smooth CN. With the D3 CN they match to about 1e-6, because the hard 40-bohr CN cutoff is discontinuous; this is not a FourierATM error.

**float32.** Only the short-range kernel runs in float32; the trace is always float64. The short-range part cancels about 20× against the trace, so forces are off by about 2e-4 relative, against 1e-5 for the real-space term. float32 is also barely faster, because the trace dominates. Keep float64.

## 3. Performance

H200, 250-atom liquids, compiled, energy + forces + stress:

| method | time |
|---|---|
| FourierATM, defaults | ≈ 0.05 s |
| real-space ATM at 40 bohr | 0.33 s |
| real-space ATM at 25 bohr | 0.04 s |

- **Where the time goes:** the trace (complex float64 GEMMs) dominates. The Ewald real-space cutoff trades k-vectors for pairs. On the 250-atom liquids, energy + forces + stress took 0.076 s at 20 bohr, 0.058 s at 25 bohr and 0.052 s at 30 bohr (the default).
- **Scaling:** the dense trace is O(N_q N³). The grid shrinks as the box grows (N_q ∝ 1/N), so at fixed accuracy the cost grows as about N². Measured scaling: see `docs/benchmark_fourier.md` (`scripts/benchmark_fourier.py`).

## 4. Settings (`FourierATMSettings`)

| field | default | meaning |
|---|---|---|
| `coordination` | `smooth_cut` | FourierD3's CN (6 Å), or `d3` for the exact D3 CN (40 bohr) |
| `short_range_cutoff_bohr`, `short_range_width_bohr` | 20, 3 | exact-damping region and its switch |
| `cp_rank` | 2 | terms of the separable C9 (max relative error 2–4e-3; higher ranks barely help) |
| `ewald_real_cutoff_bohr`, `ewald_accuracy` | 30, 1e-10 | Ewald split |
| `q_grid`, `supercell_length_bohr` | None, 70 | explicit grid, or the grid rule |

## 5. Known limitations

- **Memory and scaling:** O(N²) memory and about O(N²) time. For N ≳ several thousand atoms the real-space term is the better choice.
- **Label consistency:** differences from the 40-bohr labels are about 0.3% (the tail beyond 40 bohr plus the undamped far region).
- **Rarely-fitted species:** for large species sets the CP fit is worse (11 species: 1.6e-2 max relative error). This matters little, because it only enters the long-range part.
