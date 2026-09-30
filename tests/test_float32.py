"""float32 accuracy against the float64 oracle (design §8: document the achieved error;
target 1e-4 relative in energy). The per-triangle math runs in float32, the energy and the
gradient accumulation in float64."""

import numpy as np
import pytest
from conftest import (
    CUTOFF_SETS,
    DEVICES,
    make_settings,
    names,
    run,
    stored_reference,
    structures,
    to_atomic_units,
)


def relative_errors(name: str, device: str, compile: bool) -> tuple[float, float, float]:
    ref = stored_reference(name, "default")
    c = CUTOFF_SETS["default"]
    settings = make_settings(
        device,
        compile=compile,
        dtype="float32",
        three_body_cutoff_bohr=c["disp3"],
        coordination_cutoff_bohr=c["cn"],
    )
    got = to_atomic_units(run(structures()[name], settings))
    e_ref = float(ref["atm_energy"])
    de = abs(got.energy - e_ref) / abs(e_ref)
    df = np.abs(got.gradient - ref["atm_gradient"]).max() / np.abs(ref["atm_gradient"]).max()
    dv = 0.0
    if got.virial is not None:
        dv = np.abs(got.virial - ref["atm_virial"]).max() / np.abs(ref["atm_virial"]).max()
    print(f"float32 {name}: energy {de:.1e}, gradient {df:.1e}, virial {dv:.1e} (relative)")
    return de, df, dv


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", names("monomer") + ["dmso_crystal", "glycerol_crystal"])
def test_float32_small(name, device):
    de, df, dv = relative_errors(name, device, compile=False)
    assert de < 1e-4
    assert df < 1e-3 and dv < 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize("name", names("liquid") + names("crystal"))
def test_float32_stored(name, compile, device):
    de, df, dv = relative_errors(name, device, compile)
    assert de < 1e-4
    assert df < 1e-3 and dv < 1e-3
