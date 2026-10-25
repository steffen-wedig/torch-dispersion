"""The core package does not import torch-sim (the optional ``torchsim`` extra)."""

import subprocess
import sys


def test_core_import_does_not_load_torch_sim():
    code = (
        "import sys, torch_dispersion, torch_dispersion.calculators, torch_dispersion.d4; "
        "assert 'torch_sim' not in sys.modules, 'torch_sim imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
