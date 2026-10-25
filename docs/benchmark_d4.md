# D4 benchmark (H200)

`scripts/benchmark_d4.py` on one H200, job 712428 (`sbatch scripts/slurm/benchmark_d4.sbatch
--oracle`). It uses r2SCAN-D4 at the default cutoffs (two-body 60, ATM 40, CN 30 bohr), float64,
compiled kernels and nvalchemiops neighbour lists, and reports the median of 5 runs (2 for the
scaling rows). The columns:

- *E*: energy only.
- *EFS*: energy, forces and stress.
- *no ATM*: EFS with s9 = 0, i.e. CN, EEQ and the two-body term.
- *EEQ*: the charges alone, forward and forward + backward through the solve.
- *dftd4*: the CPU oracle (dftd4 4.1.0 wheel) for energy and gradient.

| system | atoms | E (s) | EFS (s) | no ATM (s) | EEQ q (s) | EEQ q + grad (s) | EFS peak (GiB) | dftd4 CPU (s) |
|---|---|---|---|---|---|---|---|---|
| benzamide liquid | 256 | 0.104 | 0.374 | 0.059 | 0.007 | 0.029 | 0.92 | 6.88 |
| bromobenzene liquid | 252 | 0.077 | 0.270 | 0.044 | 0.006 | 0.013 | 0.80 | 4.05 |
| cyclohexane liquid | 252 | 0.143 | 0.497 | 0.054 | 0.004 | 0.011 | 0.97 | 7.20 |
| dmso liquid | 250 | 0.097 | 0.332 | 0.040 | 0.004 | 0.010 | 0.87 | 4.39 |
| emim_ntf2 liquid | 238 | 0.082 | 0.277 | 0.046 | 0.004 | 0.010 | 0.79 | 2.89 |
| glycerol liquid | 252 | 0.142 | 0.491 | 0.039 | 0.004 | 0.028 | 0.94 | 6.67 |
| benzamide liquid ×2 | 512 | 0.193 | 0.896 | 0.085 | 0.010 | 0.037 | 1.48 | – |
| benzamide liquid ×4 | 1024 | 0.395 | 2.752 | 0.184 | 0.029 | 0.093 | 2.74 | – |
| benzamide liquid ×8 | 2048 | 0.841 | 11.707 | 0.541 | 0.105 | 0.381 | 5.94 | – |

- **The ATM gradient dominates.** The difference between EFS and no ATM is the ATM triple stage
  with gradients. It grows from ~0.3 s at 250 atoms to ~11 s at 2048, like the D3 ATM at 40 bohr
  (`docs/benchmark_fourier.md`).
- **EEQ is not a bottleneck.** The autograd EEQ gradient (solve plus the chunked Ewald direct sum)
  costs 0.38 s at 2048 atoms, about 3 % of EFS, so it is kept on autograd.
- The two-body term, CN and EEQ together take 0.04–0.06 s for the liquids.
