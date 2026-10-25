"""The examples run end to end (slow: the quick start takes ~20 s on a CPU)."""

import subprocess
import sys
from pathlib import Path

import pytest
from conftest import DEVICES

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
def test_quickstart(device):
    out = subprocess.run(
        [sys.executable, str(EXAMPLES / "quickstart.py"), "--device", device],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "water box (81 atoms), D4" in out and "step  25" in out
