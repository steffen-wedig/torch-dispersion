# Benchmark

NVIDIA H200, torch 2.14.0+cu130, kernel shape B = 32, T = 512. The six fixture liquid boxes (238-256 atoms), CN cutoff 40 bohr; median of repeated calls after a warm-up (compilation excluded). Oracle: simple-dftd3 1.6.0 on one CPU core of the same node, energy + gradient + virial.

| cutoff3 (bohr) | dtype | mode | output | time per box (s), mean [min, max] | peak GPU memory (MiB) | triangles / s | oracle (s) |
|---|---|---|---|---|---|---|---|
| 25 | float32 | compiled | E | 0.011 [0.009, 0.013] | 261 | 4.91e+09 | 3.1 |
| 25 | float32 | compiled | E + F + S | 0.031 [0.020, 0.049] | 416 | 1.82e+09 | 3.1 |
| 25 | float32 | eager | E | 0.083 [0.063, 0.122] | 1000 | 6.60e+08 | 3.1 |
| 25 | float32 | eager | E + F + S | 0.186 [0.137, 0.263] | 1267 | 2.93e+08 | 3.1 |
| 25 | float64 | compiled | E | 0.015 [0.012, 0.019] | 280 | 3.69e+09 | 3.1 |
| 25 | float64 | compiled | E + F + S | 0.040 [0.030, 0.056] | 655 | 1.37e+09 | 3.1 |
| 25 | float64 | eager | E | 0.119 [0.090, 0.176] | 1683 | 4.64e+08 | 3.1 |
| 25 | float64 | eager | E + F + S | 0.278 [0.205, 0.399] | 2352 | 1.96e+08 | 3.1 |
| 30 | float32 | compiled | E | 0.018 [0.012, 0.024] | 261 | 8.89e+09 | 10.0 |
| 30 | float32 | compiled | E + F + S | 0.058 [0.035, 0.080] | 435 | 2.83e+09 | 10.0 |
| 30 | float32 | eager | E | 0.205 [0.121, 0.296] | 1019 | 8.08e+08 | 10.0 |
| 30 | float32 | eager | E + F + S | 0.447 [0.261, 0.634] | 1286 | 3.70e+08 | 10.0 |
| 30 | float64 | compiled | E | 0.028 [0.018, 0.038] | 280 | 5.90e+09 | 10.0 |
| 30 | float64 | compiled | E + F + S | 0.091 [0.054, 0.128] | 682 | 1.82e+09 | 10.0 |
| 30 | float64 | eager | E | 0.297 [0.174, 0.429] | 1709 | 5.58e+08 | 10.0 |
| 30 | float64 | eager | E + F + S | 0.679 [0.395, 0.968] | 2377 | 2.44e+08 | 10.0 |
| 40 | float32 | compiled | E | 0.053 [0.039, 0.072] | 261 | 1.71e+10 | 36.0 |
| 40 | float32 | compiled | E + F + S | 0.205 [0.142, 0.275] | 487 | 4.45e+09 | 36.0 |
| 40 | float32 | eager | E | 0.772 [0.545, 1.068] | 1070 | 1.18e+09 | 36.0 |
| 40 | float32 | eager | E + F + S | 1.665 [1.170, 2.291] | 1338 | 5.47e+08 | 36.0 |
| 40 | float64 | compiled | E | 0.087 [0.063, 0.119] | 283 | 1.04e+10 | 36.0 |
| 40 | float64 | compiled | E + F + S | 0.328 [0.231, 0.450] | 747 | 2.77e+09 | 36.0 |
| 40 | float64 | eager | E | 1.121 [0.792, 1.552] | 1774 | 8.14e+08 | 36.0 |
| 40 | float64 | eager | E + F + S | 2.536 [1.786, 3.498] | 2442 | 3.59e+08 | 36.0 |

Triangles per box (each counted once):

| liquid | atoms | 25 bohr | 30 bohr | 40 bohr |
|---|---|---|---|---|
| benzamide_liquid | 256 | 5.09e+07 | 1.53e+08 | 8.63e+08 |
| bromobenzene_liquid | 252 | 3.51e+07 | 1.06e+08 | 5.96e+08 |
| cyclohexane_liquid | 252 | 8.24e+07 | 2.47e+08 | 1.39e+09 |
| dmso_liquid | 250 | 4.95e+07 | 1.49e+08 | 8.40e+08 |
| emim_ntf2_liquid | 238 | 3.16e+07 | 9.56e+07 | 5.40e+08 |
| glycerol_liquid | 252 | 8.13e+07 | 2.43e+08 | 1.37e+09 |

## Reading the numbers

- **Full cutoff (40 bohr), float64, compiled:** 0.33 s per 250-atom box for energy, forces
  and stress (0.09 s energy only), against 36 s for the single-threaded CPU oracle: about 110×.
- **Shortened cutoff:** 25 bohr costs 0.04 s (float64) / 0.03 s (float32) per step with forces
  and stress, about 8× less than 40 bohr, since the triangle count grows roughly as cutoff⁶
  (9.3e8 → 5.5e7 triangles on average, a factor 17 ≈ 1.6⁶). Design §9: going from 40 to 25 bohr shifts box
  energies by −6 … −14 meV (1–2 %) and forces by ≤ 0.14 meV/Å. With
  `three_body_cutoff_width_bohr > 0` the shortened term is smooth for MD.
- **float32** is 1.3–1.6× faster than float64 when compiled, at ≤ 5e-7 relative energy error
  and ≤ 1e-5 relative force error on all fixture boxes (tests/test_float32.py).
- **Compiled vs eager:** 8–13× (energy) and 7–8× (with gradients). Eager is the debugging path.
- **Memory** stays below 0.8 GiB compiled at any cutoff: it is bounded by the kernel shape
  (B × T × T), not by the number of triangles.
- Gradients cost 2.7–3.8× the energy. They come from hand-written derivatives; the
  `torch.func.vjp` kernel (`kernel_gradient="vjp"`, the reference) is ~4× slower per tile.
- Kernel shape sweep (dmso box, 40 bohr, float64, compiled, E + F + S): B × T = 16 × 256:
  0.79 s; 16 × 512: 0.34 s; 32 × 512: 0.30 s; 64 × 512: 0.28 s at 1.6× the memory. Small shapes
  are bound by per-call overhead. The defaults are B = 32, T = 512.
- First-call compilation is excluded above (warm-up call).
