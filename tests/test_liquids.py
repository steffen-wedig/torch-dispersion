"""T4: full-cutoff liquid boxes (238-256 atoms) against the stored references."""

import pytest
from conftest import (
    BACKENDS,
    COMPILE,
    CUTOFF_SETS,
    DEVICES,
    assert_matches,
    make_settings,
    names,
    oracle,
    run,
    stored_reference,
    structures,
)

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compile", COMPILE)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cutoff_set", ["default", "disp3_25"])
@pytest.mark.parametrize("name", names("liquid"))
def test_liquid(name, cutoff_set, backend, compile, device):
    c = CUTOFF_SETS[cutoff_set]
    settings = make_settings(
        device,
        backend,
        compile,
        three_body_cutoff_bohr=c["disp3"],
        coordination_cutoff_bohr=c["cn"],
    )
    # design §8: 1e-10 absolute or 1e-10 relative for large boxes
    assert_matches(run(structures()[name], settings), stored_reference(name, cutoff_set), 1e-10)


def test_stored_references_reproduce_oracle():
    """Provenance: the stored ATM energy equals a fresh oracle run. (The FHI-aims-printed D3
    totals are not part of the fixtures, so the aims cross-check of design §8 T4 is not
    repeated here; it was verified to 1e-9 eV at export.)"""
    name = "dmso_liquid"
    ref = stored_reference(name, "default")
    fresh = oracle(structures()[name], grad=False)
    assert abs(float(fresh["atm_energy"]) - float(ref["atm_energy"])) < 1e-12
    assert abs(float(fresh["two_body_energy"]) - float(ref["two_body_energy"])) < 1e-12
