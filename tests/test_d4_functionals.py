"""``D4Settings.from_functional``: the frozen ``d4.bj-eeq-atm`` table against dftd4 4.3.0's
``get_damping_param`` (every entry), and r2SCAN-D4 energies against dftd4's own lookup
(``DampingParam(method=...)``)."""

import hashlib
import json
from functools import cache

import dftd4.parameters
import numpy as np
import pytest
from ase import Atoms
from conftest import (
    TOL_ENERGY,
    UNITS,
    _d4_key,
    _d4_model,
    d4_cutoff_settings,
    run_d4,
    structures,
    to_atomic_units,
)

from torch_dispersion.settings import D4_FUNCTIONALS_PATH, D4Settings

# update together with scripts/export_d4_functionals.py
CHECKSUM = "a07bbd55df8cc3670fd61d60a572effd"
PARAMS = ("s6", "s8", "s9", "a1", "a2", "alp")
CPU = {"device": "cpu", "compile": False, **UNITS}


def table_checksum() -> str:
    table = json.loads(D4_FUNCTIONALS_PATH.read_text())["functionals"]
    return hashlib.sha256(json.dumps(table, sort_keys=True).encode()).hexdigest()[:32]


def test_checksum():
    assert table_checksum() == CHECKSUM
    assert json.loads(D4_FUNCTIONALS_PATH.read_text())["dftd4_version"] == "4.3.0"


def test_names():
    names = D4Settings.functionals()
    assert len(names) == 118 and names == sorted(names)
    assert {"r2scan", "r2scan-3c", "wb97x-2008", "pbe0_2", "dftb_3ob", "b2plyp"} <= set(names)


@pytest.mark.parametrize("name", D4Settings.functionals())
def test_parameters_match_dftd4(name):
    settings = D4Settings.from_functional(name)
    ref = dftd4.parameters.get_damping_param(name)
    assert set(ref) == set(PARAMS)
    assert {k: getattr(settings, k) for k in PARAMS} == ref


def test_case_insensitive():
    assert D4Settings.from_functional("r2SCAN") == D4Settings.from_functional("r2scan")
    assert D4Settings.from_functional("R2SCAN-3C") == D4Settings.from_functional("r2scan-3c")


def test_unknown_name():
    with pytest.raises(ValueError, match="r2scan"):
        D4Settings.from_functional("r2scan3c")  # hyphens are not ignored
    with pytest.raises(ValueError, match="available"):
        D4Settings.from_functional("not-a-functional")


def test_atm_and_overrides():
    s = D4Settings.from_functional("b2plyp", atm=False)
    assert (s.s6, s.s9, s.s8) == (0.64, 0.0, 1.16888646)
    s = D4Settings.from_functional(
        "r2scan", atm=False, s9=0.5, device="cpu", compile=False, two_body_cutoff_bohr=30.0
    )
    assert (s.s9, s.device, s.compile, s.two_body_cutoff_bohr) == (0.5, "cpu", False, 30.0)
    assert s.a1 == 0.51559235
    with pytest.raises(ValueError):  # pydantic validation of the overrides
        D4Settings.from_functional("r2scan", not_a_field=1.0)


@cache
def _method_oracle(numbers, positions, cell, charge, method: str, atm: bool, cutoffs: tuple):
    from dftd4.interface import DampingParam

    model = _d4_model(numbers, positions, cell, charge)
    model.set_realspace_cutoff(*cutoffs)
    param = DampingParam(method=method, atm=atm)
    return float(model.get_dispersion(param, grad=False)["energy"])


def two_atom_cell() -> Atoms:
    return Atoms(
        "CO", positions=[[0, 0, 0], [1.3, 0.4, 0.2]], cell=np.diag([3.2, 3.6, 3.4]), pbc=True
    )


# (disp2, disp3, cn) in bohr: dftd4's defaults for the molecule, shortened for the cell
SYSTEMS = {
    "benzamide": (lambda: structures()["benzamide_monomer_C7H7NO"], (60.0, 40.0, 30.0)),
    "two_atom_cell": (two_atom_cell, (30.0, 20.0, 20.0)),
}


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_r2scan_energy(system):
    """r2SCAN-D4 with and without ATM against dftd4's Fortran parameter lookup; ATM off leaves
    the two-body energy unchanged."""
    make, cutoffs = SYSTEMS[system]
    atoms = make()
    results = {}
    for atm in (True, False):
        settings = D4Settings.from_functional(
            "r2scan", atm=atm, **d4_cutoff_settings(*cutoffs), **CPU
        )
        results[atm] = run_d4(atoms, settings, compute_forces=False, compute_stress=False)
        ref = _method_oracle(*_d4_key(atoms, 0.0), "r2scan", atm, cutoffs)
        energy = to_atomic_units(results[atm]).energy
        assert abs(energy - ref) <= TOL_ENERGY, (atm, energy, ref)
    assert float(results[False].three_body_energy) == 0.0
    assert float(results[True].three_body_energy) != 0.0
    assert float(results[False].two_body_energy) == float(results[True].two_body_energy)
