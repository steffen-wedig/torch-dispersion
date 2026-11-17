import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import read

from torch_dispersion.atm import ATMResult, atm_dispersion
from torch_dispersion.settings import ATMSettings, BruteForceNeighbors, NvalchemiNeighbors

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
DAMPING = MANIFEST["damping"]
# the fixtures were generated with the FHI-aims constants (design §4)
BOHR = 0.52917721
HARTREE = 27.211384500
UNITS = {"bohr_in_angstrom": BOHR, "hartree_in_ev": HARTREE}
CUTOFF_SETS = MANIFEST["cutoff_sets_bohr"]

# float64 tolerances in atomic units (design §8)
TOL_ENERGY = 1e-10
TOL_GRADIENT = 1e-9
TOL_VIRIAL = 1e-9

HAS_CUDA = torch.cuda.is_available()

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
BACKENDS = ["brute_force", "nvalchemi"]
COMPILE = [False, True]


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords and not HAS_CUDA:
            item.add_marker(skip_gpu)


@cache
def structures() -> dict[str, Atoms]:
    return {a.info["name"]: a for a in read(FIXTURES / "structures.extxyz", ":")}


@cache
def _references() -> dict[str, np.ndarray]:
    return dict(np.load(FIXTURES / "references.npz"))


def stored_reference(name: str, cutoff_set: str) -> dict[str, np.ndarray]:
    refs = _references()
    prefix = f"{name}/{cutoff_set}/"
    return {k[len(prefix) :]: v for k, v in refs.items() if k.startswith(prefix)}


def backend_settings(backend: str):
    if backend == "brute_force":
        return BruteForceNeighbors()
    return NvalchemiNeighbors()


def make_settings(device="cpu", backend="brute_force", compile=False, **overrides) -> ATMSettings:
    kwargs = {
        "device": device,
        "compile": compile,
        "neighbor_backend": backend_settings(backend),
        "s9": DAMPING["s9"],
        "alpha": DAMPING["alpha"],
        **UNITS,
    }
    kwargs.update(overrides)
    return ATMSettings(**kwargs)


def run(atoms: Atoms, settings: ATMSettings, **kwargs) -> ATMResult:
    pbc = bool(atoms.pbc.all())
    return atm_dispersion(
        torch.tensor(atoms.numbers),
        torch.tensor(atoms.positions),
        torch.tensor(atoms.cell.array) if pbc else None,
        pbc,
        settings,
        **kwargs,
    )


@dataclass
class AtomicUnits:
    """An ATMResult converted to the oracle's units (Hartree, Hartree/bohr, Hartree)."""

    energy: float
    gradient: np.ndarray | None
    virial: np.ndarray | None


def to_atomic_units(result) -> AtomicUnits:
    """An ATMResult (or any result with energy, forces, virial) in atomic units."""
    return AtomicUnits(
        energy=float(result.energy) / HARTREE,
        gradient=None if result.forces is None else -result.forces.cpu().numpy() / (HARTREE / BOHR),
        virial=None if result.virial is None else result.virial.cpu().numpy() / HARTREE,
    )


def assert_matches(result: ATMResult, ref: dict, rtol_energy: float = 0.0, check_grad=True):
    got = to_atomic_units(result)
    e_ref = float(ref["atm_energy"])
    assert abs(got.energy - e_ref) <= max(TOL_ENERGY, rtol_energy * abs(e_ref)), (got.energy, e_ref)
    if check_grad and got.gradient is not None:
        np.testing.assert_allclose(got.gradient, ref["atm_gradient"], rtol=0, atol=TOL_GRADIENT)
    if check_grad and got.virial is not None and "atm_virial" in ref:
        np.testing.assert_allclose(got.virial, ref["atm_virial"], rtol=0, atol=TOL_VIRIAL)


@cache
def _oracle(
    numbers: tuple, positions: tuple, cell: tuple | None, cutoffs: tuple, grad: bool
) -> dict[str, np.ndarray]:
    from dftd3.interface import DispersionModel, RationalDampingParam

    pos = np.array(positions).reshape(-1, 3) / BOHR
    lattice = None if cell is None else np.array(cell).reshape(3, 3) / BOHR
    periodic = None if cell is None else np.array([True, True, True])
    model = DispersionModel(np.array(numbers), pos, lattice, periodic)
    model.set_realspace_cutoff(*cutoffs)

    def param(s9):
        d = DAMPING
        return RationalDampingParam(
            s6=d["s6"], s8=d["s8"], s9=s9, a1=d["a1"], a2=d["a2"], alp=d["alpha"]
        )

    full = model.get_dispersion(param(1.0), grad=grad)
    two = model.get_dispersion(param(0.0), grad=grad)
    out = {"atm_energy": np.array(full["energy"] - two["energy"])}
    out["two_body_energy"] = np.array(two["energy"])
    if grad:
        out["atm_gradient"] = full["gradient"] - two["gradient"]
        if cell is not None:
            out["atm_virial"] = full["virial"] - two["virial"]
    return out


def oracle(
    atoms: Atoms,
    disp3: float = 40.0,
    cn: float = 40.0,
    width3: float = 0.0,
    disp2: float = 60.0,
    grad: bool = True,
) -> dict[str, np.ndarray]:
    """ATM reference = get_dispersion(s9 = 1) - get_dispersion(s9 = 0), atomic units."""
    pbc = bool(atoms.pbc.all())
    return _oracle(
        tuple(atoms.numbers.tolist()),
        tuple(atoms.positions.reshape(-1).tolist()),
        tuple(atoms.cell.array.reshape(-1).tolist()) if pbc else None,
        (disp2, disp3, cn, 0.0, width3),
        grad,
    )


def names(kind: str) -> list[str]:
    return [n for n in structures() if f"_{kind}" in n]


def random_cluster(n: int, seed: int, numbers=(1, 6, 7, 8, 9, 16, 35), density=0.08) -> Atoms:
    """Random cluster with a 1.0 Å minimum distance (close contacts exercise the damping)."""
    rng = np.random.default_rng(seed)
    box = (n / density) ** (1 / 3)
    pos: list[np.ndarray] = []
    while len(pos) < n:
        x = rng.uniform(0, box, 3)
        if all(np.linalg.norm(x - p) > 1.0 for p in pos):
            pos.append(x)
    z = np.array([numbers[i % len(numbers)] for i in range(n)])
    return Atoms(numbers=z, positions=np.array(pos))


def random_cell(n: int, seed: int, cell_lengths=(4.5, 5.0, 5.5), angles=(80, 95, 105)) -> Atoms:
    rng = np.random.default_rng(seed)
    from ase.geometry import cellpar_to_cell

    cell = cellpar_to_cell([*cell_lengths, *angles])
    while True:
        frac = rng.uniform(0, 1, (n, 3))
        atoms = Atoms(
            numbers=[(1, 6, 7, 8)[i % 4] for i in range(n)],
            scaled_positions=frac,
            cell=cell,
            pbc=True,
        )
        d = atoms.get_all_distances(mic=True)
        if n == 1 or d[np.triu_indices(n, 1)].min() > 1.0:
            return atoms


# cell parameters (Å, degrees) of the sparse test boxes
BOXES = {
    "orthorhombic": (16.0, 17.0, 18.0, 90.0, 90.0, 90.0),
    "triclinic": (16.0, 17.0, 18.0, 70.0, 80.0, 110.0),
    "hexagonal": (16.0, 16.0, 18.0, 90.0, 90.0, 120.0),
    "elongated": (13.0, 16.0, 26.0, 90.0, 90.0, 90.0),
    "skewed": (15.0, 16.0, 17.0, 60.0, 65.0, 70.0),
}


def monomer_in_box(name="benzamide_monomer_C7H7NO", box="orthorhombic") -> Atoms:
    """A fixture monomer centred in one of ``BOXES``: periodic but sparse, so the brute-force
    triangle sum converges at affordable cutoffs."""
    from ase.geometry import cellpar_to_cell

    atoms = structures()[name].copy()
    atoms.cell = cellpar_to_cell(BOXES[box])
    atoms.pbc = True
    atoms.positions += atoms.cell.array.sum(0) / 2 - atoms.positions.mean(0)
    return atoms


# --------------------------------------------------------------------------------------------
# DFT-D4 (oracle: dftd4 4.3.0; default cutoffs 60 / 40 / 30 bohr for disp2 / disp3 / CN)

# r2SCAN-D4 (dftd4 parameters.toml, d4.bj-eeq-atm)
D4_PARAMS = {
    "s6": 1.0,
    "s8": 0.60187490,
    "s9": 1.0,
    "a1": 0.51559235,
    "a2": 5.77342911,
    "alp": 16.0,
}
D4_DEFAULT_CUTOFFS = (60.0, 40.0, 30.0)
# the fixture ions (formal charges) for the charged-system tests
ION_CHARGES = {"emim_ntf2_monomer_C6H11N2": 1.0, "emim_ntf2_monomer_C2F6NO4S2": -1.0}


def make_d4_settings(device="cpu", backend="brute_force", compile=False, **overrides):
    from torch_dispersion.settings import D4Settings

    kwargs = {
        "device": device,
        "compile": compile,
        "neighbor_backend": backend_settings(backend),
        **D4_PARAMS,
        **UNITS,
    }
    kwargs.update(overrides)
    return D4Settings(**kwargs)


def d4_cutoff_settings(disp2: float, disp3: float, cn: float) -> dict[str, float]:
    """``D4Settings`` overrides matching ``d4_oracle(..., cutoffs=(disp2, disp3, cn))``."""
    return {
        "two_body_cutoff_bohr": disp2,
        "three_body_cutoff_bohr": disp3,
        "coordination_cutoff_bohr": cn,
    }


def run_d4(atoms: Atoms, settings, charge: float = 0.0, **kwargs):
    from torch_dispersion.d4 import d4_dispersion

    pbc = bool(atoms.pbc.all())
    return d4_dispersion(
        torch.tensor(atoms.numbers),
        torch.tensor(atoms.positions),
        torch.tensor(atoms.cell.array) if pbc else None,
        pbc,
        settings,
        charge=charge,
        **kwargs,
    )


def _d4_model(numbers: tuple, positions: tuple, cell: tuple | None, charge: float):
    from dftd4.interface import DispersionModel

    pos = np.array(positions).reshape(-1, 3) / BOHR
    lattice = None if cell is None else np.array(cell).reshape(3, 3) / BOHR
    periodic = None if cell is None else np.array([True, True, True])
    return DispersionModel(np.array(numbers), pos, charge, lattice, periodic)


def _d4_key(atoms: Atoms, charge: float) -> tuple:
    pbc = bool(atoms.pbc.all())
    return (
        tuple(atoms.numbers.tolist()),
        tuple(atoms.positions.reshape(-1).tolist()),
        tuple(atoms.cell.array.reshape(-1).tolist()) if pbc else None,
        float(charge),
    )


@cache
def _d4_oracle(numbers, positions, cell, charge, params: tuple, cutoffs: tuple, grad: bool):
    from dftd4.interface import DampingParam

    model = _d4_model(numbers, positions, cell, charge)
    model.set_realspace_cutoff(*cutoffs)
    res = model.get_dispersion(DampingParam(**dict(params)), grad=grad)
    return {k: np.array(v) for k, v in res.items()}


def d4_oracle(
    atoms: Atoms,
    charge: float = 0.0,
    grad: bool = True,
    cutoffs: tuple[float, float, float] = D4_DEFAULT_CUTOFFS,
    **params,
):
    """dftd4 energy (Hartree), gradient (Hartree/bohr) and virial (Hartree) at the real-space
    cutoffs (disp2, disp3, cn) in bohr."""
    full = {**D4_PARAMS, **params}
    return _d4_oracle(*_d4_key(atoms, charge), tuple(sorted(full.items())), cutoffs, grad)


@cache
def _d4_properties(numbers, positions, cell, charge):
    return _d4_model(numbers, positions, cell, charge).get_properties()


def d4_properties(atoms: Atoms, charge: float = 0.0) -> dict[str, np.ndarray]:
    """CN, charges, C6 and polarisabilities at the default cutoffs."""
    return _d4_properties(*_d4_key(atoms, charge))
