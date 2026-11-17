"""ORCA's D4rev functionals, wB97X-D4rev and wB97M-D4rev (keywords ``WB97X-D4REV``,
``WB97M-D4REV``, ORCA >= 6.0).

D4rev is not a different model. It is dftd4's ``d4`` model (EEQ-2019 charges, BJ two-body,
ATM with s9 = 1) with revised damping parameters. ORCA 6.1.1 manual, section 3.4.1.7,
Table 3.15, "DFT-D4 (!D4) parameters used for various functionals" lists them, and dftd4
4.3.0 ships the same numbers as ``wb97x-rev`` (doi 10.1063/5.0133026) and ``wb97m-rev``
(doi 10.1021/acs.jctc.3c00717). ``D4Settings.from_functional("wb97m-rev")`` is
therefore wB97M-D4rev.

Two checks:
- the frozen table against ORCA's published numbers, copied literally below;
- energy, gradient and virial against dftd4 4.3.0, which looks up the parameters by name
  in its own Fortran table (``DampingParam(method=...)``, independent of our JSON table).
  The checks cover molecules, ions, a cluster of main-group, transition-metal and heavy
  elements, crystals, ATM on and off, compiled kernels, CUDA and a TorchSim batch.
"""

from functools import cache

import numpy as np
import pytest
import torch
from conftest import (
    BOHR,
    COMPILE,
    DEVICES,
    HARTREE,
    ION_CHARGES,
    TOL_ENERGY,
    TOL_GRADIENT,
    TOL_VIRIAL,
    UNITS,
    _d4_key,
    _d4_model,
    d4_cutoff_settings,
    names,
    random_cluster,
    run_d4,
    structures,
    to_atomic_units,
)

from torch_dispersion.settings import D4Settings

# ORCA 6.1.1 manual, Table 3.15: (s6, s8, a1, a2, s9)
ORCA_D4REV = {
    "wb97x-rev": (1.0000, 0.44850000, 0.33060000, 4.27900000, 1.0000),
    "wb97m-rev": (1.0000, 0.84200000, 0.35900000, 4.66800000, 1.0000),
}
FUNCTIONALS = list(ORCA_D4REV)
DEFAULT = (60.0, 40.0, 30.0)  # dftd4's real-space cutoffs (disp2, disp3, cn) in bohr
SHORT = (30.0, 20.0, 20.0)
# main group, transition metals and heavy elements, as in OMol25-like chemistry
WIDE_ELEMENTS = (1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 26, 29, 30, 35, 46, 53, 78)


def settings(name: str, device="cpu", compile=False, atm=True, cutoffs=DEFAULT, **kw) -> D4Settings:
    return D4Settings.from_functional(
        name, atm=atm, device=device, compile=compile, **d4_cutoff_settings(*cutoffs), **UNITS, **kw
    )


@cache
def _oracle(numbers, positions, cell, charge, method: str, atm: bool, cutoffs: tuple):
    from dftd4.interface import DampingParam

    model = _d4_model(numbers, positions, cell, charge)
    model.set_realspace_cutoff(*cutoffs)
    res = model.get_dispersion(DampingParam(method=method, atm=atm), grad=True)
    return {k: np.array(v) for k, v in res.items()}


def oracle(atoms, method: str, charge=0.0, atm=True, cutoffs=DEFAULT):
    """dftd4 4.3.0 with its own parameters for ``method``, atomic units."""
    return _oracle(*_d4_key(atoms, charge), method, atm, cutoffs)


def assert_matches(result, ref):
    got = to_atomic_units(result)
    assert abs(got.energy - float(ref["energy"])) <= TOL_ENERGY, (got.energy, ref["energy"])
    np.testing.assert_allclose(got.gradient, ref["gradient"], rtol=0, atol=TOL_GRADIENT)
    if got.virial is not None:
        np.testing.assert_allclose(got.virial, ref["virial"], rtol=0, atol=TOL_VIRIAL)


@pytest.mark.parametrize("name", FUNCTIONALS)
def test_parameters_match_orca(name):
    s6, s8, a1, a2, s9 = ORCA_D4REV[name]
    s = D4Settings.from_functional(name)
    assert (s.s6, s.s8, s.a1, s.a2, s.s9) == (s6, s8, a1, a2, s9)
    assert s.alp == 16.0  # not in ORCA's table; dftd4's default, as for every D4 functional


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compile", COMPILE)
@pytest.mark.parametrize("molecule", names("monomer"))
@pytest.mark.parametrize("name", FUNCTIONALS)
def test_molecules(name, molecule, compile, device):
    atoms = structures()[molecule]
    charge = ION_CHARGES.get(molecule, 0.0)
    result = run_d4(atoms, settings(name, device, compile), charge=charge)
    assert_matches(result, oracle(atoms, name, charge=charge))
    assert float(result.three_body_energy) != 0.0


@pytest.mark.parametrize("molecule", names("monomer"))
@pytest.mark.parametrize("name", FUNCTIONALS)
def test_atm_off(name, molecule):
    atoms = structures()[molecule]
    charge = ION_CHARGES.get(molecule, 0.0)
    result = run_d4(atoms, settings(name, atm=False), charge=charge)
    assert_matches(result, oracle(atoms, name, charge=charge, atm=False))
    assert float(result.three_body_energy) == 0.0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("charge", [0.0, 2.0])
@pytest.mark.parametrize("name", FUNCTIONALS)
def test_wide_element_cluster(name, charge, device):
    """80 atoms of 21 elements up to Pt, 1 Å close contacts, neutral and dication."""
    atoms = random_cluster(80, seed=11, numbers=WIDE_ELEMENTS)
    result = run_d4(atoms, settings(name, device), charge=charge)
    assert_matches(result, oracle(atoms, name, charge=charge))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("crystal", names("crystal"))
@pytest.mark.parametrize("name", FUNCTIONALS)
def test_crystals_short_cutoff(name, crystal, device):
    atoms = structures()[crystal]
    result = run_d4(atoms, settings(name, device, cutoffs=SHORT))
    assert_matches(result, oracle(atoms, name, cutoffs=SHORT))


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("crystal", names("crystal"))
@pytest.mark.parametrize("name", FUNCTIONALS)
def test_crystals(name, crystal, device):
    atoms = structures()[crystal]
    assert_matches(run_d4(atoms, settings(name, device)), oracle(atoms, name))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", FUNCTIONALS)
def test_torchsim_molecule_batch(name, device):
    """The batch-wide TorchSim pipeline on molecules and ions against dftd4 per molecule."""
    ts = pytest.importorskip("torch_sim")
    from torch_sim.typing import SystemExtras

    from torch_dispersion.calculators.torchsim import D4Model

    batch = []
    for molecule in names("monomer"):
        atoms = structures()[molecule].copy()
        atoms.info["charge"] = ION_CHARGES.get(molecule, 0.0)
        batch.append(atoms)
    cluster = random_cluster(80, seed=11, numbers=WIDE_ELEMENTS)
    cluster.info["charge"] = 2.0
    batch.append(cluster)
    state = ts.io.atoms_to_state(
        batch,
        torch.device(device),
        torch.float64,
        system_extras_map={SystemExtras.TOTAL_CHARGE: "charge"},
    )
    out = D4Model(settings(name, device), compute_stress=False)(state)
    forces = out["forces"].cpu().numpy()
    start = 0
    for i, atoms in enumerate(batch):
        ref = oracle(atoms, name, charge=atoms.info["charge"])
        energy = float(out["energy"][i]) / HARTREE
        assert abs(energy - float(ref["energy"])) <= TOL_ENERGY, (i, energy, ref["energy"])
        gradient = -forces[start : start + len(atoms)] / (HARTREE / BOHR)
        np.testing.assert_allclose(gradient, ref["gradient"], rtol=0, atol=TOL_GRADIENT)
        start += len(atoms)
