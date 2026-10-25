"""The batched TorchSim model ``torch_dispersion.calculators.torchsim.D4Model``.

Every system of a batch must equal ``d4_dispersion`` on that system alone (energy, forces,
stress, charges): exactly with the brute-force backend on the CPU (same per-system lists, same
code path), to rounding with nvalchemiops (the batched list orders neighbours differently) and
on CUDA (atomic scatter-adds are not deterministic, not even between two calls). Shortened
cutoffs keep the CPU cases fast; the ~1000-atom mixed batches with ATM or brute force are slow.
"""

import numpy as np
import pytest
import torch
from conftest import DEVICES, ION_CHARGES, make_d4_settings, random_cell, run_d4, structures
from torch._dynamo.utils import counters

ts = pytest.importorskip("torch_sim")

from torch_sim.models.interface import SumModel, validate_model_outputs  # noqa: E402
from torch_sim.models.soft_sphere import SoftSphereModel  # noqa: E402
from torch_sim.typing import SystemExtras  # noqa: E402

from torch_dispersion.calculators.torchsim import D4Model  # noqa: E402

SHORT = {
    "two_body_cutoff_bohr": 25.0,
    "three_body_cutoff_bohr": 15.0,
    "coordination_cutoff_bohr": 18.0,
}


def rattled(atoms, seed: int, amplitude: float = 0.05):
    atoms = atoms.copy()
    atoms.positions += np.random.default_rng(seed).normal(0.0, amplitude, atoms.positions.shape)
    return atoms


def equal_cells():
    """Three cells of the same size (one crystal, rattled)."""
    return [rattled(structures()["dmso_crystal"], seed) for seed in range(3)]


def mixed_cells():
    """Small cells and a ~1000-atom box: 10, 3, 252, 1024 and 34 atoms."""
    return [
        structures()["dmso_crystal"],
        random_cell(3, seed=3),
        structures()["cyclohexane_liquid"],
        structures()["benzamide_liquid"].repeat((2, 2, 1)),
        structures()["emim_ntf2_crystal"],
    ]


def molecules():
    """Neutral monomers and the two ions (total charges in ``atoms.info``)."""
    out = []
    for name in ("benzamide_monomer_C7H7NO", *ION_CHARGES, "dmso_monomer_C2H6OS"):
        atoms = structures()[name].copy()
        atoms.info["charge"] = ION_CHARGES.get(name, 0.0)
        out.append(atoms)
    return out


BATCHES = {"equal": equal_cells, "mixed": mixed_cells, "molecules": molecules}


def to_state(atoms_list, device):
    return ts.io.atoms_to_state(
        atoms_list,
        torch.device(device),
        torch.float64,
        system_extras_map={SystemExtras.TOTAL_CHARGE: "charge"},
    )


def assert_matches_loop(out, atoms_list, settings, compute_charges: bool, exact: bool):
    """``out`` equals a loop of single-system ``d4_dispersion`` calls."""
    refs = [run_d4(a, settings, charge=a.info.get("charge", 0.0)) for a in atoms_list]
    pbc = bool(atoms_list[0].pbc.all())
    expected = {
        "energy": torch.stack([r.energy for r in refs]),
        "forces": torch.cat([r.forces for r in refs]),
    }
    if pbc:
        expected["stress"] = torch.stack([r.stress for r in refs])
    if compute_charges:
        expected["charges"] = torch.cat([r.charges for r in refs])
    assert set(out) == set(expected)
    for key, value in expected.items():
        assert out[key].dtype == torch.float64
        assert_same(out[key], value, exact)


def assert_same(got, expected, exact: bool, atol: float = 1e-12):
    """Bit for bit if ``exact`` on the CPU, else to rounding. On CUDA the batched and single
    nvalchemiops lists order neighbours differently and the scatters are atomic; the EEQ solve
    amplifies that rounding to ~3e-13 (charges in e, forces in eV/Å)."""
    if exact and got.device.type == "cpu":
        assert torch.equal(got, expected), (got - expected).abs().max()
    else:
        torch.testing.assert_close(got, expected, rtol=1e-12, atol=atol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["brute_force", "nvalchemi"])
def test_validate_model_outputs(backend, device):
    model = D4Model(make_d4_settings(device, backend=backend, **SHORT))
    validate_model_outputs(model, torch.device(device), torch.float64)


def _batch_params():
    params = []
    for batch in BATCHES:
        for backend in ("brute_force", "nvalchemi"):
            for s9 in (0.0, 1.0):
                heavy = batch == "mixed" and (s9 != 0.0 or backend == "brute_force")
                marks = [pytest.mark.slow] if heavy else []
                params.append(pytest.param(batch, backend, s9, marks=marks))
    return params


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("batch", "backend", "s9"), _batch_params())
def test_batch_equals_loop(batch, backend, s9, device):
    atoms_list = BATCHES[batch]()
    settings = make_d4_settings(device, backend=backend, s9=s9, **SHORT)
    model = D4Model(settings, compute_charges=True)
    out = model(to_state(atoms_list, device))
    assert_matches_loop(out, atoms_list, settings, True, exact=backend == "brute_force")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["brute_force", "nvalchemi"])
@pytest.mark.parametrize("batch", ["equal", "molecules"])
def test_per_system_neighbor_lists(batch, backend, device):
    """``batch_neighbor_list=False`` (lists built system by system) equals the loop too."""
    atoms_list = BATCHES[batch]()
    settings = make_d4_settings(device, backend=backend, **SHORT)
    model = D4Model(settings, compute_charges=True, batch_neighbor_list=False)
    out = model(to_state(atoms_list, device))
    assert_matches_loop(out, atoms_list, settings, True, exact=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compute_charges", [False, True])
def test_charges_output(compute_charges, device):
    """Charges only on request; the ions keep their total charge."""
    atoms_list = molecules()
    settings = make_d4_settings(device, **SHORT)
    out = D4Model(settings, compute_charges=compute_charges)(to_state(atoms_list, device))
    assert ("charges" in out) == compute_charges
    assert_matches_loop(out, atoms_list, settings, compute_charges, exact=True)
    if compute_charges:
        totals = [float(c.sum()) for c in out["charges"].split([len(a) for a in atoms_list])]
        np.testing.assert_allclose(totals, [a.info["charge"] for a in atoms_list], atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_energy_only_and_attributes(device):
    settings = make_d4_settings(device, **SHORT)
    model = D4Model(settings, compute_forces=False, compute_stress=False)
    assert model.memory_scales_with == "n_atoms_x_density"
    assert model.device == settings.torch_device and model.dtype == torch.float64
    atoms_list = equal_cells()
    out = model(to_state(atoms_list, device))
    assert set(out) == {"energy"}
    expected = torch.stack([run_d4(a, settings).energy for a in atoms_list])
    assert_same(out["energy"], expected, exact=True)
    model.compute_forces = model.compute_stress = True
    assert set(model(to_state(atoms_list, device))) == {"energy", "forces", "stress"}


@pytest.mark.parametrize("device", DEVICES)
def test_positions_requiring_grad(device):
    """Positions with requires_grad (as in a differentiable workflow) are accepted, and the
    outputs are detached and equal."""
    model = D4Model(make_d4_settings(device, **SHORT))
    state = to_state(equal_cells(), device)
    ref = model(state)
    state.positions.requires_grad_(True)
    out = model(state)
    for key, value in ref.items():
        assert not out[key].requires_grad
        assert_same(out[key], value, exact=True)


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_pbc_raises(device):
    state = to_state(equal_cells(), device)
    state.pbc = torch.tensor([True, True, False], device=device)
    with pytest.raises(ValueError, match="mixed periodicity"):
        D4Model(make_d4_settings(device, **SHORT))(state)


@pytest.mark.parametrize("device", DEVICES)
def test_sum_model_optimize(device):
    """SumModel(soft-sphere repulsion, D4) through a few LBFGS steps with the Frechet cell
    filter: the sum is additive, and the relaxation moves atoms and cells."""
    atoms_list = [structures()["dmso_crystal"], rattled(structures()["benzamide_crystal"], 1)]
    soft = SoftSphereModel(
        sigma=1.5, epsilon=0.05, device=torch.device(device), compute_stress=True
    )
    d4 = D4Model(make_d4_settings(device, **SHORT))
    model = SumModel(soft, d4)
    assert model.compute_forces and model.compute_stress
    state = to_state(atoms_list, device)
    out, out_soft, out_d4 = model(state), soft(state), d4(state)
    for key in ("energy", "forces", "stress"):
        torch.testing.assert_close(out[key], out_soft[key] + out_d4[key], rtol=1e-14, atol=1e-15)

    final = ts.optimize(
        system=state,
        model=model,
        optimizer=ts.Optimizer.lbfgs,
        max_steps=4,
        steps_between_swaps=2,
        init_kwargs={"cell_filter": ts.CellFilter.frechet},
    )
    assert final.n_systems == 2
    assert bool(torch.isfinite(final.energy).all())
    assert not torch.allclose(final.positions, state.positions)
    assert not torch.allclose(final.cell, state.cell)


def unique_graphs() -> int:
    return counters["stats"]["unique_graphs"]


@pytest.mark.parametrize("device", DEVICES)
def test_new_batch_does_not_recompile(device):
    """After warm-up, batches of other sizes, compositions and system counts compile no new
    graph: each system goes through the same dynamic-shape graph stage and fixed-shape kernels
    as ``d4_dispersion`` (``test_d4_compile.test_new_system_size_does_not_recompile``)."""
    model = D4Model(make_d4_settings(device, compile=True, **SHORT))
    model(to_state([random_cell(4, seed=1), random_cell(5, seed=2)], device))
    model(to_state([random_cell(6, seed=6)], device))
    before = unique_graphs()
    for sizes in [(3, 7), (11,), (4, 6, 9)]:
        batch = [
            random_cell(n, seed=n, cell_lengths=(4.0 + n / 2, 4.5 + n / 2, 5.0 + n / 2))
            for n in sizes
        ]
        model(to_state(batch, device))
    assert unique_graphs() == before
