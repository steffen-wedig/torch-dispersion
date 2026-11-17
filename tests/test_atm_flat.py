"""The flat-batch ATM kernel (``atm.flat``) against the per-system kernel (``atm.kernel``).

The flat kernel drops the far side's derivatives and scales the centre legs by 3/2
(``docs/notes/atm_pair_gradients.md``), so its forces, virials and dE/dC6 are summed differently
from the per-system kernel's but equal to rounding. dE/dC6 is compared symmetrised: the two
kernels split the derivative of the symmetric C6 between (i, j) and (j, i) differently. The
batches mix system sizes, so blocks get different tile widths and tiles span two systems.
"""

import pytest
import torch
from conftest import DEVICES, random_cell, random_cluster

from torch_dispersion.atm.flat import c6_tables, flat_inputs, flat_scalars, get_flat_step_fn
from torch_dispersion.atm.kernel import KernelParams, get_step_fn, triple_inputs
from torch_dispersion.core.batch import Batch
from torch_dispersion.core.padding import pad_positions, padded_size
from torch_dispersion.core.tiles import (
    Accumulators,
    SliceLayout,
    TileLayout,
    run_tiles,
    zero_energy_or_accumulators,
)
from torch_dispersion.d4.data import d4_reference_data
from torch_dispersion.neighbors import build_flat
from torch_dispersion.pairwise.rational import rational_tables
from torch_dispersion.settings import BruteForceNeighbors

F64 = torch.float64
BLOCK, TILE, MIN_TILE = 8, 64, 16
CUTOFF, ALPHA, S9 = 18.0, 14.0, 1.0


def batch_inputs(atoms_list, device):
    """Positions (bohr), cells, numbers, counts, neighbour list and random C6 blocks."""
    counts = [len(a) for a in atoms_list]
    pbc = bool(atoms_list[0].pbc.all())
    pos = torch.cat([torch.tensor(a.positions) for a in atoms_list]).to(device) / 0.52917721
    cells = None
    if pbc:
        cells = torch.stack([torch.tensor(a.cell.array) for a in atoms_list]).to(device)
        cells = cells / 0.52917721
    numbers = torch.cat([torch.tensor(a.numbers) for a in atoms_list]).to(device)
    nl = build_flat(BruteForceNeighbors(), pos, cells, pbc, CUTOFF, counts)
    n_max = max(counts)
    g = torch.Generator().manual_seed(0)
    c6 = 5.0 + 40.0 * torch.rand(len(counts), n_max, n_max, generator=g, dtype=F64)
    c6 = (0.5 * (c6 + c6.mT)).to(device)
    return pos, cells, numbers, counts, nl, c6


def per_system(pos, cells, numbers, counts, nl, c6, r0, with_grad):
    """The previous D4 path: one per-system kernel run per system."""
    out = []
    start = 0
    for s, n in enumerate(counts):
        n_padded = padded_size(n, BLOCK)
        nl_s = nl.rows(start, n).pad(n_padded, TILE)
        p = pad_positions(pos[start : start + n], n_padded)
        z = torch.nn.functional.pad(numbers[start : start + n], (0, n_padded - n))
        cell = None if cells is None else cells[s]
        layout = TileLayout.build(nl_s, nl_s.vectors(p, cell), z, BLOCK, TILE)
        c6_p = torch.nn.functional.pad(c6[s, :n, :n], (0, n_padded - n, 0, n_padded - n))
        scalars = KernelParams(s9=S9, alpha=ALPHA, cutoff=CUTOFF, width=0.0)
        out.append(
            run_tiles(
                get_step_fn(with_grad, False),
                zero_energy_or_accumulators(with_grad, n_padded, pos.device),
                (c6_p,),
                layout,
                n,
                2,
                triple_inputs(r0),
                scalars.scalars(F64, pos.device),
            )
        )
        start += n
    return out


def flat(pos, cells, numbers, counts, nl, c6, r0, with_grad, compile=False):
    device = pos.device
    batch = Batch.from_counts(counts, device)
    n = batch.n_atoms
    n_padded = padded_size(n, BLOCK)
    pad = (0, n_padded - n)
    system = torch.nn.functional.pad(batch.system, pad)
    rows = pos.new_zeros(n_padded, 3, 3)
    if cells is not None:
        rows[:n] = cells[batch.system]
    layout = SliceLayout.build(nl, n_padded, BLOCK, TILE, MIN_TILE)
    sqrt_c6, inv_c6 = c6_tables(c6.reshape(-1))
    tables = (
        sqrt_c6,
        inv_c6,
        pad_positions(pos, n_padded),
        rows,
        torch.nn.functional.pad(numbers, pad),
        torch.nn.functional.pad(batch.local, pad),
    )
    acc = zero_energy_or_accumulators(with_grad, n_padded, device, len(counts), c6.numel())
    scalars = flat_scalars(S9, ALPHA, CUTOFF, 0.0, batch.n_max, F64, device)
    acc = run_tiles(
        get_flat_step_fn(with_grad, compile, TILE), acc, tables, layout, n, 2,
        flat_inputs(r0, system), scalars,
    )  # fmt: skip
    return acc, layout


def molecules():
    return [random_cluster(n, seed=n) for n in (3, 9, 30, 14, 2, 21)]


def cells():
    return [random_cell(n, seed=n) for n in (1, 2, 5, 3)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["molecules", "cells"])
def test_flat_matches_per_system(kind, device):
    atoms_list = molecules() if kind == "molecules" else cells()
    pos, cell, numbers, counts, nl, c6 = batch_inputs(atoms_list, device)
    _, r0 = rational_tables(d4_reference_data(torch.device(device), F64).r4r2, 0.4, 4.5)
    refs = per_system(pos, cell, numbers, counts, nl, c6, r0, True)
    acc, layout = flat(pos, cell, numbers, counts, nl, c6, r0, True)
    assert isinstance(acc, Accumulators)
    assert layout.widths is not None
    if kind == "molecules":
        assert len(set(layout.widths)) > 1, "several tile widths"
    else:
        assert max(layout.counts) > TILE, "several tiles per row"

    energy = torch.stack([r.energy for r in refs])
    has_triangles = torch.tensor(counts, device=energy.device) >= 3
    assert ((energy.abs() > 0) == has_triangles).all() or kind == "cells"
    torch.testing.assert_close(acc.energy, energy, rtol=1e-13, atol=0)
    grad_pos = torch.cat([r.grad_pos[:n] for r, n in zip(refs, counts, strict=True)])
    torch.testing.assert_close(acc.grad_pos[: sum(counts)], grad_pos, rtol=1e-11, atol=1e-16)
    if cell is not None:
        virial = torch.stack([r.virial for r in refs])
        torch.testing.assert_close(acc.virial, virial, rtol=1e-11, atol=1e-16)
    n_max = max(counts)
    g_flat = acc.grad_c6.view(len(counts), n_max, n_max)
    for s, (r, n) in enumerate(zip(refs, counts, strict=True)):
        g_ref = r.grad_c6[:n, :n]
        g = g_flat[s, :n, :n]
        torch.testing.assert_close(g + g.T, g_ref + g_ref.T, rtol=1e-11, atol=1e-18)

    # the energy-only step agrees with the gradient step
    energy_only, _ = flat(pos, cell, numbers, counts, nl, c6, r0, False)
    torch.testing.assert_close(energy_only, acc.energy, rtol=1e-14, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_flat_compiled_matches_eager(device):
    pos, cell, numbers, counts, nl, c6 = batch_inputs(cells(), device)
    _, r0 = rational_tables(d4_reference_data(torch.device(device), F64).r4r2, 0.4, 4.5)
    eager, _ = flat(pos, cell, numbers, counts, nl, c6, r0, True)
    compiled, _ = flat(pos, cell, numbers, counts, nl, c6, r0, True, compile=True)
    for a, b in zip(eager, compiled, strict=True):
        torch.testing.assert_close(b, a, rtol=1e-12, atol=1e-18)
