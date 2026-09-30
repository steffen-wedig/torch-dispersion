"""T9: torch.compile.

Recompilation control: the triple kernel has static shapes fixed by the settings
(centre_block_size x neighbor_tile_size), the only system-sized input (C6) has dynamic
dimensions, and the parameters are 0-d tensors; stage 1 (CN, C6) is compiled with dynamic shapes.
"""

import numpy as np
import pytest
import torch
from conftest import DEVICES, make_settings, random_cell, run, structures
from torch._dynamo.utils import counters

import torch_atm.kernel as kernel
from torch_atm.kernel import KernelParams, TileInputs

SHORT = {"three_body_cutoff_bohr": 20.0, "coordination_cutoff_bohr": 20.0}


def unique_graphs() -> int:
    return counters["stats"]["unique_graphs"]


def dummy_tile(device, dtype=torch.float64, block=4, tile=16, n=9):
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, n, (block, tile), generator=g).to(device)
    mask = torch.rand(block, tile, generator=g).to(device) > 0.2
    inputs = TileInputs(
        centre=torch.arange(block, device=device),
        idx_n=idx,
        idx_m=idx.flip(-1),
        mask_n=mask,
        mask_m=mask.flip(-1),
        z_centre=torch.full((block,), 6, device=device),
        z_n=torch.full((block, tile), 1, device=device),
        z_m=torch.full((block, tile), 8, device=device),
        r0=torch.rand(104, 104, generator=g, dtype=dtype).to(device) * 4 + 4,
        diagonal=torch.tensor(False, device=device),
    )
    v_n = torch.randn(block, tile, 3, generator=g, dtype=dtype).to(device) * 5
    v_m = torch.randn(block, tile, 3, generator=g, dtype=dtype).to(device) * 5
    c6 = torch.rand(n, n, generator=g, dtype=dtype).to(device) * 30 + 1
    c6 = 0.5 * (c6 + c6.T)
    scalars = KernelParams(1.0, 14.0, 20.0, 0.0).scalars(dtype, torch.device(device))
    return v_n, v_m, c6, inputs, scalars


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("with_grad", [False, True])
def test_kernel_fullgraph(with_grad, device):
    """(a) fullgraph=True compiles (a graph break would raise), and matches eager."""
    args = dummy_tile(device)
    compiled = kernel.get_tile_fn(with_grad, compile=True)(*args)
    eager = kernel.get_tile_fn(with_grad, compile=False)(*args)
    for c, e in zip(
        compiled if with_grad else [compiled], eager if with_grad else [eager], strict=True
    ):
        torch.testing.assert_close(c, e, rtol=1e-13, atol=1e-16)


@pytest.mark.parametrize("device", DEVICES)
def test_vjp_matches_autograd(device):
    """(c) no analytic kernel exists yet: torch.func.vjp against plain torch.autograd."""
    v_n, v_m, c6, inputs, scalars = dummy_tile(device)
    e, g_n, g_m, g_c6 = kernel.tile_energy_and_grad(v_n, v_m, c6, inputs, scalars)
    leaves = [x.clone().requires_grad_() for x in (v_n, v_m, c6)]
    e2 = kernel.tile_energy(*leaves, inputs, scalars)
    grads = torch.autograd.grad(e2, leaves)
    torch.testing.assert_close(e, e2, rtol=0, atol=0)
    for a, b in zip((g_n, g_m, g_c6), grads, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["dmso_crystal", "glycerol_monomer_C3H8O3", "fcc_ar"])
def test_compiled_equals_eager(name, device):
    """(b) Compiled vs eager in float64. Inductor reorders reductions, so this is not bit-for-bit;
    the observed differences are at the 1e-16 relative level."""
    from ase.build import bulk

    atoms = bulk("Ar", "fcc", a=5.26) if name == "fcc_ar" else structures()[name]
    eager = run(atoms, make_settings(device, compile=False, **SHORT))
    compiled = run(atoms, make_settings(device, compile=True, **SHORT))
    torch.testing.assert_close(compiled.energy, eager.energy, rtol=1e-14, atol=0)
    torch.testing.assert_close(compiled.forces, eager.forces, rtol=0, atol=1e-14)
    if eager.stress is not None:
        torch.testing.assert_close(compiled.virial, eager.virial, rtol=0, atol=1e-13)


@pytest.mark.parametrize("device", DEVICES)
def test_no_recompiles_in_md(device):
    """(d) 50 MD-like steps (small displacements) trigger zero recompilations."""
    atoms = random_cell(6, seed=21)
    settings = make_settings(device, compile=True, **SHORT)
    run(atoms, settings)  # warm-up (compiles or reuses)
    rng = np.random.default_rng(0)
    with torch._dynamo.config.patch(error_on_recompile=True):
        before = unique_graphs()
        for _ in range(50):
            atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
            run(atoms, settings)
        assert unique_graphs() == before


@pytest.mark.parametrize("device", DEVICES)
def test_system_change_does_not_recompile_kernel(device):
    """(e) A different system (other N, list width, species) reuses the kernel graph; stage 1 is
    dynamic, so at most one new graph appears (its first dynamic-shape recompile)."""
    settings = make_settings(device, compile=True, **SHORT)
    run(random_cell(4, seed=1), settings)
    run(random_cell(5, seed=2), settings)
    before = unique_graphs()
    run(structures()["dmso_crystal"], settings)
    run(structures()["glycerol_crystal"], settings)
    assert unique_graphs() - before <= 1


@pytest.mark.parametrize("device", DEVICES)
def test_new_tile_shape_compiles_once(device):
    """(e) A new kernel bucket (tile size) compiles exactly one new kernel graph."""
    atoms = random_cell(4, seed=1)
    run(atoms, make_settings(device, compile=True, **SHORT))
    before = unique_graphs()
    new_bucket = make_settings(device, compile=True, neighbor_tile_size=96, **SHORT)
    run(atoms, new_bucket)
    run(random_cell(5, seed=2), new_bucket)
    assert unique_graphs() - before == 1
