"""T1: the CN -> weights -> C6 chain, validated through the D3(BJ) two-body energy.

simple-dftd3 exposes no C6, so the D3 C6 of the ATM graph stage is fed to the shared
rational-damping pair kernel (``pairwise.rational``, the D4 two-body kernel) and the resulting
two-body energy is compared with the oracle's.
"""

import pytest
import torch
from conftest import (
    BACKENDS,
    DAMPING,
    DEVICES,
    TOL_ENERGY,
    make_settings,
    names,
    oracle,
    random_cluster,
    stored_reference,
    structures,
)

from torch_dispersion.atm.realspace import _stage1
from torch_dispersion.core.inputs import prepare
from torch_dispersion.core.padding import CN_WIDTH_MULTIPLE, pad_positions, padded_size
from torch_dispersion.core.tiles import TileLayout, run_tiles
from torch_dispersion.d3.data import pad_system, reference_data
from torch_dispersion.neighbors import make_builder
from torch_dispersion.pairwise.rational import (
    get_pair_step_fn,
    pair_inputs,
    pair_scalars,
    rational_tables,
)

CUTOFF = 60.0  # bohr, simple-dftd3's default two-body cutoff


def two_body(atoms, settings) -> float:
    """D3(BJ) two-body energy (Hartree) from this package's D3 C6."""
    pbc = bool(atoms.pbc.all())
    cell = torch.tensor(atoms.cell.array) if pbc else None
    inputs = prepare(
        torch.tensor(atoms.numbers), torch.tensor(atoms.positions), cell, pbc, settings
    )
    device, dtype = settings.torch_device, settings.torch_dtype
    pos, cell_b, n = inputs.positions, inputs.cell, len(atoms)
    data = reference_data(device, dtype)
    block, tile = settings.centre_block_size, settings.neighbor_tile_size
    n_padded = padded_size(n, block)
    system = pad_system(inputs.numbers, n_padded, data)
    pos_p = pad_positions(pos, n_padded)
    builder = make_builder(settings.neighbor_backend)
    ccn = settings.coordination_cutoff_bohr
    nl_cn = builder(pos, cell_b, pbc, ccn).pad(n_padded, CN_WIDTH_MULTIPLE)
    tables = (system.rcov, system.ref_cn, system.ref_mask, system.species, system.c6_species)
    _, c6 = _stage1(pos_p, cell_b, nl_cn, *tables, ccn)
    nl = builder(pos, cell_b, pbc, CUTOFF).pad(n_padded, tile)
    layout = TileLayout.build(nl, nl.vectors(pos_p, cell_b), system.numbers, block, tile)
    rr, r0 = rational_tables(data.r4r2, DAMPING["a1"], DAMPING["a2"])
    energy = run_tiles(
        get_pair_step_fn(False, compile=False),
        torch.zeros((), dtype=torch.float64, device=device),
        (c6,),
        layout,
        n,
        1,
        pair_inputs(rr, r0),
        pair_scalars(DAMPING["s6"], DAMPING["s8"], CUTOFF, dtype, device),
    )
    return float(energy)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", names("monomer"))
def test_monomers(name, backend, device):
    ref = stored_reference(name, "default")["two_body_energy"]
    got = two_body(structures()[name], make_settings(device, backend))
    assert abs(got - float(ref)) <= TOL_ENERGY


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", names("crystal"))
def test_crystals(name, backend, device):
    ref = stored_reference(name, "default")["two_body_energy"]
    got = two_body(structures()[name], make_settings(device, backend))
    assert abs(got - float(ref)) <= TOL_ENERGY


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_random_cluster_all_elements(backend, device):
    atoms = random_cluster(28, seed=1)
    assert set(atoms.numbers) == {1, 6, 7, 8, 9, 16, 35}
    ref = oracle(atoms, grad=False)["two_body_energy"]
    got = two_body(atoms, make_settings(device, backend))
    assert abs(got - float(ref)) <= TOL_ENERGY
