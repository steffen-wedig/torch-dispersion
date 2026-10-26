"""torch.compile of the D4 kernels (inductor: C++ on the CPU, Triton on CUDA).

The two-body pair kernel compiles with ``fullgraph=True`` and equals eager; its hand-written
gradients equal autograd; and a new system size does not recompile anything once the
dynamic-shape graph-stage graphs have been built (end-to-end compiled-vs-eager and the 50-step
MD test are in ``test_d4_derivatives.py``).
"""

import pytest
import torch
from conftest import BOHR, DEVICES, make_d4_settings, random_cell, run_d4
from torch._dynamo.utils import counters

from torch_dispersion.core.tiles import SliceLayout, run_tiles, zero_accumulators
from torch_dispersion.d4.data import d4_reference_data
from torch_dispersion.neighbors import brute_force_neighbors
from torch_dispersion.pairwise import rational

SHORT = {
    "two_body_cutoff_bohr": 25.0,
    "three_body_cutoff_bohr": 15.0,
    "coordination_cutoff_bohr": 18.0,
}


def unique_graphs() -> int:
    return counters["stats"]["unique_graphs"]


def dummy_tile(device, block=4, tile=16, n=9, box=30.0, cutoff=20.0):
    """Pair tile of ``block`` centres over random neighbours and lattice shifts (-1, 0, 1) of a
    ``box`` bohr cubic cell; some pairs lie beyond the cutoff, and padded entries are masked.
    Returns the kernel's per-atom tables (one system), the tile inputs and the scalars."""
    g = torch.Generator().manual_seed(0)
    f64 = torch.float64
    pos = (torch.rand(n, 3, generator=g, dtype=f64) * box).to(device)
    cell = torch.eye(3, dtype=f64, device=device) * box
    numbers = torch.tensor([1, 6, 8, 7, 1, 6, 8, 16, 35])[:n].to(device)
    centre = torch.arange(block, device=device)
    idx = torch.randint(0, n, (block, tile), generator=g).to(device)
    shift = torch.randint(-1, 2, (block, tile, 3), generator=g).to(device, torch.int8)
    self_image = (idx == centre[:, None]) & (shift == 0).all(-1)
    mask = (torch.rand(block, tile, generator=g) > 0.2).to(device) & ~self_image
    r4r2 = d4_reference_data(torch.device(device), f64).r4r2
    rr = 3.0 * r4r2[:, None] * r4r2[None, :]
    inputs = rational.PairTileInputs(
        centre=centre,
        idx=idx,
        shift=shift,
        mask=mask,
        system=torch.zeros_like(centre),
        rr=rr,
        r0=0.5 * torch.sqrt(rr) + 5.7,
    )
    c6 = (torch.rand(n, n, generator=g, dtype=f64) * 30 + 1).to(device)
    c6 = 0.5 * (c6 + c6.T)
    local = torch.arange(n, device=device)
    scalars = rational.pair_scalars(1.0, 0.6, cutoff, n, f64, torch.device(device))
    return (pos, cell, numbers, local), c6.reshape(-1), inputs, scalars


def tables(pos, cell, numbers, local):
    return pos, cell.expand(pos.shape[0], 3, 3), numbers, local


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("with_grad", [False, True])
def test_pair_kernel_fullgraph(with_grad, device):
    """fullgraph=True compiles (a graph break would raise) and matches eager."""
    atoms, c6, t, s = dummy_tile(device)
    n = atoms[0].shape[0]

    def zero():
        if with_grad:
            return zero_accumulators(n, torch.device(device), 1, c6.numel())
        return torch.zeros(1, dtype=torch.float64, device=device)

    args = (c6, *tables(*atoms), t, s)
    compiled = rational.get_pair_step_fn(with_grad, compile=True)(zero(), *args)
    eager = rational.get_pair_step_fn(with_grad, compile=False)(zero(), *args)
    compiled = compiled if with_grad else [compiled]
    eager = eager if with_grad else [eager]
    assert float(eager[0]) != 0.0
    for c, e in zip(compiled, eager, strict=True):
        torch.testing.assert_close(c, e, rtol=1e-13, atol=1e-18)


def run_pair_tiles(step, acc, c6, pos, cell, numbers, nl, block, tile, cutoff):
    """All tiles of a full neighbour list of one system (``nl`` over ``pos``) through
    ``step``."""
    n, n_padded = pos.shape[0], -(-(pos.shape[0] + 1) // block) * block
    pos_p = torch.cat([pos, pos.new_zeros(n_padded - n, 3)])
    numbers_p = torch.nn.functional.pad(numbers, (0, n_padded - n))
    rows = torch.arange(n_padded, device=pos.device)
    r4r2 = d4_reference_data(pos.device, pos.dtype).r4r2
    rr, r0 = rational.rational_tables(r4r2, 0.5, 5.7)
    return run_tiles(
        step,
        acc,
        (c6, pos_p, cell.expand(n_padded, 3, 3), numbers_p, rows),
        SliceLayout.build(nl, n_padded, block, tile),
        n,
        1,
        rational.pair_inputs(rr, r0, torch.zeros_like(rows)),
        rational.pair_scalars(1.0, 0.6, cutoff, n_padded, pos.dtype, pos.device),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compile", [False, True])
def test_pair_gradients_match_autograd(compile, device):
    """Hand-written energy, dE/dR, virial (dE/dε) and dE/dC6 of the pair kernel over all tiles
    of a full neighbour list equal autograd through the energy of strained positions and cell.
    The forces come from row sums (pair symmetry of the full list), so this needs a real list:
    a small cell with self-images, several tiles per row and several centre blocks."""
    atoms = random_cell(5, seed=3, cell_lengths=(4.0, 4.4, 4.8))
    f64, dev = torch.float64, torch.device(device)
    pos = torch.tensor(atoms.positions, dtype=f64, device=dev) / BOHR
    cell = torch.tensor(atoms.cell.array, dtype=f64, device=dev) / BOHR
    numbers = torch.tensor(atoms.numbers, device=dev)
    cutoff, block, tile = 22.0, 4, 64
    nl = brute_force_neighbors(pos, cell, True, cutoff)
    n_padded = -(-(len(atoms) + 1) // block) * block
    g = torch.Generator().manual_seed(0)
    c6 = torch.rand(n_padded, n_padded, generator=g, dtype=f64) * 30 + 1
    c6 = (0.5 * (c6 + c6.T)).reshape(-1).to(dev)

    pos_leaf, c6_leaf = pos.clone().requires_grad_(), c6.clone().requires_grad_()
    strain = torch.zeros(3, 3, dtype=f64, device=dev, requires_grad=True)
    deform = torch.eye(3, dtype=f64, device=dev) + strain
    zero = torch.zeros(1, dtype=f64, device=dev)
    energy = run_pair_tiles(
        rational.energy_step, zero, c6_leaf, pos_leaf @ deform, cell @ deform, numbers, nl,
        block, tile, cutoff,
    )  # fmt: skip
    grad_pos, grad_c6, virial = torch.autograd.grad(energy.sum(), (pos_leaf, c6_leaf, strain))

    step = rational.get_pair_step_fn(True, compile)
    acc0 = zero_accumulators(n_padded, dev, 1, c6.numel())
    acc = run_pair_tiles(step, acc0, c6, pos, cell, numbers, nl, block, tile, cutoff)
    assert nl.width > tile and n_padded > block, "several tiles and blocks"
    assert bool(grad_pos.abs().sum() > 0) and bool(grad_c6.abs().sum() > 0)
    torch.testing.assert_close(acc.energy, energy.detach(), rtol=1e-14, atol=0)
    torch.testing.assert_close(acc.grad_pos[: len(atoms)], grad_pos, rtol=1e-11, atol=1e-18)
    torch.testing.assert_close(acc.grad_c6, grad_c6, rtol=1e-12, atol=1e-20)
    torch.testing.assert_close(acc.virial, virial[None], rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("device", DEVICES)
def test_new_system_size_does_not_recompile(device):
    """After warm-up, cells with other atom counts, list widths and species compile no new
    graph: the graph-stage graphs are compiled with dynamic shapes from the start, the kernels
    have fixed tile shapes, and N is padded (``test_new_tile_shape_compiles_once`` shows the
    counter does register new graphs)."""
    settings = make_d4_settings(device, compile=True, **SHORT)
    run_d4(random_cell(4, seed=1), settings)
    run_d4(random_cell(5, seed=2), settings)
    before = unique_graphs()
    for n, lengths in [(3, (4.0, 4.5, 5.0)), (7, (6.0, 6.5, 7.0)), (11, (7.5, 8.0, 8.5))]:
        run_d4(random_cell(n, seed=n, cell_lengths=lengths), settings)
    assert unique_graphs() == before


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("field", "value", "graphs"),
    # the pair kernel is one graph, the analytic ATM kernel two (pointwise and reduction).
    # Compiled kernels are cached process-wide, so the sizes must not occur in any other test
    # (test_compile.py uses 96, test_blocking.py 32, 64 and 256; the default is 512).
    [("pair_tile_size", 1024, 1), ("neighbor_tile_size", 80, 2)],
)
def test_new_tile_shape_compiles_once(field, value, graphs, device):
    """A new kernel bucket compiles that kernel once and nothing else, for any system."""
    atoms = random_cell(4, seed=1)
    run_d4(atoms, make_d4_settings(device, compile=True, **SHORT))
    before = unique_graphs()
    settings = make_d4_settings(device, compile=True, **SHORT, **{field: value})
    run_d4(atoms, settings)
    run_d4(random_cell(6, seed=6), settings)
    assert unique_graphs() - before == graphs
