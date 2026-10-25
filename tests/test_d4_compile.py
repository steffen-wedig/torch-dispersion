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
    """Pair tile of ``block`` centres over random neighbours in a ``box`` bohr cube (no PBC);
    some pairs lie beyond the cutoff, and padded entries are masked."""
    g = torch.Generator().manual_seed(0)
    f64 = torch.float64
    pos = (torch.rand(n, 3, generator=g, dtype=f64) * box).to(device)
    numbers = torch.tensor([1, 6, 8, 7, 1, 6, 8, 16, 35])[:n].to(device)
    centre = torch.arange(block, device=device)
    idx = torch.randint(0, n, (block, tile), generator=g).to(device)
    mask = (torch.rand(block, tile, generator=g) > 0.2).to(device) & (idx != centre[:, None])
    r4r2 = d4_reference_data(torch.device(device), f64).r4r2
    rr = 3.0 * r4r2[:, None] * r4r2[None, :]
    inputs = rational.PairTileInputs(
        centre=centre,
        idx=idx,
        mask=mask,
        z_centre=numbers[centre],
        z=numbers[idx],
        rr=rr,
        r0=0.5 * torch.sqrt(rr) + 5.7,
    )
    c6 = (torch.rand(n, n, generator=g, dtype=f64) * 30 + 1).to(device)
    c6 = 0.5 * (c6 + c6.T)
    scalars = rational.pair_scalars(1.0, 0.6, cutoff, f64, torch.device(device))
    return pos, c6, inputs, scalars


def vectors(pos, t):
    return torch.where(t.mask[..., None], pos[t.idx] - pos[t.centre][:, None, :], 0.0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("with_grad", [False, True])
def test_pair_kernel_fullgraph(with_grad, device):
    """fullgraph=True compiles (a graph break would raise) and matches eager."""
    pos, c6, t, s = dummy_tile(device)
    v = vectors(pos, t)

    def zero():
        if with_grad:
            return zero_accumulators(c6.shape[0], torch.device(device))
        return torch.zeros((), dtype=torch.float64, device=device)

    compiled = rational.get_pair_step_fn(with_grad, compile=True)(zero(), v, c6, t, s)
    eager = rational.get_pair_step_fn(with_grad, compile=False)(zero(), v, c6, t, s)
    compiled = compiled if with_grad else [compiled]
    eager = eager if with_grad else [eager]
    assert float(eager[0]) != 0.0
    for c, e in zip(compiled, eager, strict=True):
        torch.testing.assert_close(c, e, rtol=1e-13, atol=1e-18)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compile", [False, True])
def test_pair_gradients_match_autograd(compile, device):
    """Hand-written energy, dE/dR, virial and dE/dC6 of the pair kernel equal autograd."""
    pos, c6, t, s = dummy_tile(device)
    pos_leaf, c6_leaf = pos.clone().requires_grad_(), c6.clone().requires_grad_()
    v = vectors(pos_leaf, t)
    zero = torch.zeros((), dtype=torch.float64, device=device)
    energy = rational.energy_step(zero, v, c6_leaf, t, s)
    grad_pos, grad_c6, grad_v = torch.autograd.grad(energy, (pos_leaf, c6_leaf, v))
    virial = torch.einsum("btx,bty->xy", v.detach(), grad_v)

    step = rational.get_pair_step_fn(True, compile)
    acc = step(zero_accumulators(c6.shape[0], torch.device(device)), v.detach(), c6, t, s)
    assert bool(grad_pos.abs().sum() > 0) and bool(grad_c6.abs().sum() > 0)
    torch.testing.assert_close(acc.energy, energy.detach(), rtol=1e-14, atol=0)
    torch.testing.assert_close(acc.grad_pos, grad_pos, rtol=1e-12, atol=1e-20)
    torch.testing.assert_close(acc.grad_c6, grad_c6, rtol=1e-12, atol=1e-20)
    torch.testing.assert_close(acc.virial, virial, rtol=1e-12, atol=1e-20)


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
