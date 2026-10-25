"""Export the D4 rational-damping parameters per functional into
``src/torch_dispersion/d4/functionals.json``.

Source (LGPL-3.0): dftd4's ``parameters.toml``, shipped in the ``dftd4`` wheel (the pinned dev
dependency). Each functional's ``d4.bj-eeq-atm`` entry is merged onto
``[default.parameter] d4.bj-eeq-atm`` (s6 = 1, s9 = 1, alp = 16), as
``dftd4.parameters.get_damping_param`` does, and only the numeric parameters are kept (no
DOIs, no damping / mbd tags). ``D4Settings.from_functional`` reads the result.

Run with ``uv run python scripts/export_d4_functionals.py``; the checksum in
``tests/test_d4_functionals.py`` must be updated whenever the output changes.
"""

import hashlib
import importlib.metadata
import json
import tomllib
from pathlib import Path

import dftd4.parameters

OUT = Path(__file__).resolve().parents[1] / "src" / "torch_dispersion" / "d4" / "functionals.json"
SCHEME = "bj-eeq-atm"
KEYS = ("s6", "s8", "s9", "a1", "a2", "alp")


def main() -> None:
    raw = Path(dftd4.parameters.get_data_file_name()).read_bytes()
    table = tomllib.loads(raw.decode())
    default = table["default"]["parameter"]["d4"][SCHEME]
    functionals = {}
    for name, entry in table["parameter"].items():
        merged = {**default, **entry["d4"][SCHEME]}
        functionals[name] = {k: float(merged[k]) for k in KEYS}
    header = {
        "source": f"dftd4 parameters.toml, d4.{SCHEME} merged onto [default.parameter]",
        "dftd4_version": importlib.metadata.version("dftd4"),
        "parameters_toml_sha256": hashlib.sha256(raw).hexdigest(),
    }
    # one line per functional
    lines = [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in header.items()]
    body = ",\n".join(f"    {json.dumps(k)}: {json.dumps(v)}" for k, v in functionals.items())
    lines.append(f'  "functionals": {{\n{body}\n  }}')
    OUT.write_text("{\n" + ",\n".join(lines) + "\n}\n")
    print(f"wrote {OUT} ({len(functionals)} functionals)")


if __name__ == "__main__":
    main()
