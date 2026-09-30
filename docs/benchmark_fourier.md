# FourierATM benchmark

NVIDIA H200, torch 2.14.0+cu130, SLURM job 514952 (`sbatch scripts/slurm/benchmark_fourier.sbatch`, `scripts/benchmark_fourier.py`).

Setup:

- float64, compiled;
- median of 5 calls after a warm-up (compilation and the C9 fit excluded);
- FourierATM uses the default settings: r_s = 20 bohr, Ewald real cutoff 30 bohr, rank-2 C9, 3³ grid for the liquids, analytic trace gradients.

## Liquids

Energy differences are in meV/atom against the real-space D3 at 80 bohr (D3 CN). The 80-bohr sum still lacks about +0.0013 meV/atom of tail. The smooth-CN column also contains the effect of changing the CN, so it is not a pure truncation error.

| liquid | atoms | FourierATM (smooth CN) | FourierATM (D3 CN) | real 40 bohr | real 25 bohr | FourierATM E / E+F+S (s) | real 40 E+F+S (s) | real 25 E+F+S (s) |
|---|---|---|---|---|---|---|---|---|
| benzamide | 256 | +0.0003 | -0.0025 | -0.0091 | -0.0462 | 0.019 / 0.050 | 0.312 | 0.033 |
| bromobenzene | 252 | -0.0003 | -0.0044 | -0.0137 | -0.0681 | 0.023 / 0.048 | 0.231 | 0.032 |
| cyclohexane | 252 | +0.0002 | -0.0007 | -0.0065 | -0.0317 | 0.024 / 0.069 | 0.450 | 0.056 |
| dmso | 250 | -0.0010 | -0.0018 | -0.0093 | -0.0464 | 0.019 / 0.049 | 0.296 | 0.030 |
| emim_ntf2 | 238 | -0.0001 | -0.0014 | -0.0064 | -0.0317 | 0.019 / 0.051 | 0.232 | 0.031 |
| glycerol | 252 | +0.0011 | +0.0003 | -0.0066 | -0.0318 | 0.024 / 0.071 | 0.450 | 0.056 |

## Scaling

The benzamide liquid replicated 1–8×, energy + forces + stress, with peak GPU memory in brackets:

| atoms | FourierATM analytic | FourierATM autograd | real 25 bohr | real 40 bohr |
|---|---|---|---|---|
| 256 | 0.053 s (2.5 GiB) | 0.054 s (2.5 GiB) | 0.033 s (0.6 GiB) | 0.313 s (0.7 GiB) |
| 512 | 0.124 s (2.5 GiB) | 0.158 s (5.3 GiB) | 0.082 s (0.9 GiB) | 0.802 s (1.0 GiB) |
| 1024 | 0.416 s (4.2 GiB) | 0.629 s (10.5 GiB) | 0.254 s (1.5 GiB) | 2.548 s (1.9 GiB) |
| 2048 | 1.905 s (13.9 GiB) | 3.019 s (27.7 GiB) | 1.060 s (3.3 GiB) | 11.082 s (4.0 GiB) |

## Observations

- **Against real space at 40 bohr:** FourierATM is 6× faster at every size up to 2048 atoms, and 3–10× closer to the converged value (D3 CN column).
- **Against real space at 25 bohr:** that is 1.6–1.8× faster than FourierATM, but 10–70× less accurate.
- **Analytic trace gradients:** on par with autograd at 256 atoms and 1.3–1.6× faster above that. Peak memory halves from 512 atoms.
- **Scaling:** the grid rule drops to 2³ at 2048 atoms, so the cost grows as about N² (0.05 → 1.9 s for 8× the atoms).
- **Real-space kernel at 40 bohr:** it also grows faster than linearly, 36× for 8× the atoms. Its C6 matrix and its one-hot scatters are O(N) per tile.
