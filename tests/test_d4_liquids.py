"""D4 on the fixture liquid boxes (238-256 atoms) against dftd4 4.3.0 at the default cutoffs
(slow: the CPU oracle takes ~10 s per evaluation; run on a GPU)."""

import pytest
from conftest import DEVICES, d4_oracle, make_d4_settings, names, run_d4, structures
from test_d4_dispersion import assert_matches

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", names("liquid"))
def test_liquids(name, device):
    atoms = structures()[name]
    result = run_d4(atoms, make_d4_settings(device, compile=device == "cuda"))
    assert_matches(result, d4_oracle(atoms), rtol_energy=1e-10)
