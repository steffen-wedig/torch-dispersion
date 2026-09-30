import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import read

from torch_atm.atm import ATMResult, atm_dispersion
from torch_atm.settings import ATMSettings, BruteForceNeighbors, NvalchemiNeighbors

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
try:
    import nvalchemiops  # noqa: F401

    HAS_NVALCHEMI = True
except ImportError:
    HAS_NVALCHEMI = False

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
    if not HAS_NVALCHEMI:
        pytest.skip("nvalchemiops not installed")
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


def to_atomic_units(result: ATMResult) -> AtomicUnits:
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
