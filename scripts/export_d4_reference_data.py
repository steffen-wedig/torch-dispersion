"""Export the D4 reference tables into ``src/torch_dispersion/d4/reference_data.npz``.

Sources (Apache-2.0): tad-dftd4 (reference CNs, charges, polarisabilities, Gaussian-weight
counts, effective charges, chemical hardnesses, r4/r2 values), tad-multicharge (EEQ-2019
parameters) and tad-mctc (Pauling electronegativities). The D3 covalent radii (4/3 scaled, bohr)
are shared with the D3 tables (``d3/reference_data.npz``). Elements Z = 0..103; row 0 is the
padding dummy without references.

Precision: tad-multicharge stores the EEQ parameters and tad-dftd4 builds r4r2 in float32, so
those values are parsed from the literal source text and evaluated in float64.

The reference polarisabilities alpha(i omega) are evaluated here, in float64, with the default
charge-scaling parameters ga = 3, gc = 2 (dftd4 ``set_refalpha_eeq``):

    alpha_ref = max(ascale (alpha_ref,raw - hcount sscale_s alpha_s(iw) zeta_s), 0)
    zeta_s = zeta(ga, gc eta_s, Z_s, q_h + Z_s)

with s the reference's secondary (hydride) system, Z_s its effective charge and q_h the
reference charge of the hydrogens.

Run with ``uv run python scripts/export_d4_reference_data.py``; the checksum in
``tests/test_d4_data.py`` must be updated whenever the output changes.
"""

import ast
import hashlib
import inspect
from pathlib import Path

import numpy as np
import tad_mctc.data.en
import tad_multicharge.param.eeq2019
import torch
from tad_dftd4.data import GAM, ZEFF
from tad_dftd4.data import r4r2 as tad_r4r2
from tad_dftd4.reference import d4 as d4ref
from tad_dftd4.reference.d4 import charge_eeq

ROOT = Path(__file__).resolve().parents[1] / "src" / "torch_dispersion"
OUT = ROOT / "d4" / "reference_data.npz"
N_Z = 104  # Z = 0..103
GA, GC = 3.0, 2.0


def _flatten(node: ast.AST) -> list[float]:
    """Numbers of a (possibly starred, nested) list literal."""
    if isinstance(node, ast.List):
        out = []
        for elt in node.elts:
            out += _flatten(elt.value if isinstance(elt, ast.Starred) else elt)
        return out
    return [float(ast.literal_eval(node))]


def _literal(module, name: str) -> torch.Tensor:
    """The list literal assigned to ``name`` in ``module`` (directly or as the first argument
    of ``torch.tensor``), parsed in float64."""
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            value = node.value
            if isinstance(value, ast.Call):
                value = value.args[0]
            return torch.tensor(_flatten(value), dtype=torch.float64)
    raise KeyError(name)


def _zeta(a: float, c: torch.Tensor, qref: torch.Tensor, qmod: torch.Tensor) -> torch.Tensor:
    safe = torch.where(qmod < 0.0, 1.0, qmod)
    return torch.where(
        qmod < 0.0,
        torch.exp(torch.tensor(a, dtype=torch.float64)),
        torch.exp(a * (1.0 - torch.exp(c * (1.0 - qref / safe)))),
    )


def main() -> None:
    f64 = torch.float64
    ngw = d4ref.refc[:N_Z].to(torch.int64)  # Gaussians per reference, 0 for missing ones
    n_ref = (ngw > 0).sum(-1)
    present = ngw > 0
    ref_cn = torch.where(present, d4ref.refcovcn[:N_Z].to(f64), 0.0)
    ref_q = torch.where(present, charge_eeq.clsq[:N_Z].to(f64), 0.0)
    zeff = ZEFF().to(f64)
    hardness = GAM(dtype=f64)

    # reference polarisabilities on the 23-point imaginary frequency grid
    refsys = d4ref.refsys[:N_Z]
    zs = zeff[refsys]
    zeta = _zeta(GA, GC * hardness[refsys], zs, charge_eeq.clsh[:N_Z].to(f64) + zs)
    sec = d4ref.secscale.to(f64)[refsys] * d4ref.secalpha.to(f64)[refsys] * zeta[..., None]
    hydride = (refsys > 0)[..., None]
    raw = d4ref.refalpha[:N_Z].to(f64) - torch.where(
        hydride, d4ref.refscount[:N_Z].to(f64)[..., None] * sec, 0.0
    )
    alpha = (d4ref.refascale[:N_Z].to(f64)[..., None] * raw).clamp_min(0.0)
    alpha = torch.where(present[..., None], alpha, 0.0)

    r4_over_r2 = _literal(tad_r4r2, "_r4_over_r2")
    r4r2 = torch.sqrt(0.5 * r4_over_r2 * torch.sqrt(torch.arange(len(r4_over_r2), dtype=f64)))

    with np.load(ROOT / "d3" / "reference_data.npz") as d3:
        rcov = torch.from_numpy(d3["covalent_radius"])
    eeq = tad_multicharge.param.eeq2019
    tables = {
        "covalent_radius": rcov,
        "electronegativity": _literal(tad_mctc.data.en, "_PAULING"),
        "zeff": zeff,
        "hardness": hardness,
        "r4r2": r4r2,
        "n_ref": n_ref,
        "n_gaussians": ngw,
        "reference_cn": ref_cn,
        "reference_q": ref_q,
        "alpha_iw": alpha,
        "eeq_chi": _literal(eeq, "chi"),
        "eeq_eta": _literal(eeq, "eta"),
        "eeq_kcnchi": _literal(eeq, "kcn"),
        "eeq_rad": _literal(eeq, "rad"),
    }
    tables = {k: v[:N_Z].contiguous() for k, v in tables.items()}
    for name, t in tables.items():
        assert t.shape[0] == N_Z, (name, t.shape)
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **{k: v.numpy() for k, v in tables.items()})
    print(f"wrote {OUT}")
    print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
