"""torch.compile of the D4 kernels (inductor: C++ on the CPU, Triton on CUDA).

The two-body pair kernel compiles with ``fullgraph=True`` and equals eager; its hand-written
gradients equal autograd; and a new system size does not recompile anything once the
dynamic-shape graph-stage graphs have been built (end-to-end compiled-vs-eager and the 50-step
MD test are in ``test_d4_derivatives.py``).
"""

import pytest
import torch
from conftest import DEVICES, make_d4_settings, random_cell, run_d4
from torch._dynamo.utils import counters

from torch_dispersion.core.tiles import zero_accumulators
from torch_dispersion.d4.data import d4_reference_data
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compile", [False, True])
def test_pair_gradients_match_autograd(compile, device):
    """Hand-written energy, dE/dR, virial (dE/dε) and dE/dC6 of the pair kernel equal autograd
    through the energy of strained positions and cell."""
    (pos, cell, numbers, local), c6, t, s = dummy_tile(device)
    pos_leaf, c6_leaf = pos.clone().requires_grad_(), c6.clone().requires_grad_()
    strain = torch.zeros(3, 3, dtype=pos.dtype, device=device, requires_grad=True)
    deform = torch.eye(3, dtype=pos.dtype, device=device) + strain
    atoms = tables(pos_leaf @ deform, cell @ deform, numbers, local)
    zero = torch.zeros(1, dtype=torch.float64, device=device)
    energy = rational.energy_step(zero, c6_leaf, *atoms, t, s)
    grad_pos, grad_c6, virial = torch.autograd.grad(energy.sum(), (pos_leaf, c6_leaf, strain))

    step = rational.get_pair_step_fn(True, compile)
    acc0 = zero_accumulators(pos.shape[0], torch.device(device), 1, c6.numel())
    acc = step(acc0, c6, *tables(pos, cell, numbers, local), t, s)
    assert bool(grad_pos.abs().sum() > 0) and bool(grad_c6.abs().sum() > 0)
    torch.testing.assert_close(acc.energy, energy.detach(), rtol=1e-14, atol=0)
    torch.testing.assert_close(acc.grad_pos, grad_pos, rtol=1e-12, atol=1e-20)
    torch.testing.assert_close(acc.grad_c6, grad_c6, rtol=1e-12, atol=1e-20)
    torch.testing.assert_close(acc.virial, virial[None], rtol=1e-12, atol=1e-20)


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
