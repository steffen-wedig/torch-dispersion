"""Summarise a benchmark JSON (scripts/benchmark.py) as Markdown tables.

    uv run python scripts/benchmark_report.py benchmarks/results/h200_<job>.json \
        > docs/benchmarks/d3_atm.md
"""

import json
import statistics
import sys
from collections import defaultdict


def main() -> None:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    rows, meta = data["rows"], data["meta"]
    print("# Benchmark\n")
    print(
        f"{meta['device']}, torch {meta['torch']}, kernel shape B = {meta['block']}, "
        f"T = {meta['tile']}. The six fixture liquid boxes (238-256 atoms), CN cutoff 40 bohr; "
        "median of repeated calls after a warm-up (compilation excluded). Oracle: "
        "simple-dftd3 1.6.0 on one CPU core of the same node, energy + gradient + virial.\n"
    )
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cutoff3_bohr"], r["dtype"], r["mode"], r["forces_stress"])].append(r)
    print(
        "| cutoff3 (bohr) | dtype | mode | output | time per box (s), mean [min, max] "
        "| peak GPU memory (MiB) | triangles / s | oracle (s) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for (cutoff, dtype, mode, grad), rs in sorted(groups.items()):
        t = [r["seconds"] for r in rs]
        oracle = [r["oracle_seconds"] for r in rs if r["oracle_seconds"]]
        oracle_cell = f"{statistics.mean(oracle):.1f}" if oracle else "-"
        print(
            f"| {cutoff:.0f} | {dtype} | {mode} | {'E + F + S' if grad else 'E'} "
            f"| {statistics.mean(t):.3f} [{min(t):.3f}, {max(t):.3f}] "
            f"| {max(r['peak_mib'] for r in rs):.0f} "
            f"| {statistics.mean(r['triangles_per_s'] for r in rs):.2e} "
            f"| {oracle_cell} |"
        )
    print("\nTriangles per box (each counted once):\n")
    print(
        "| liquid | atoms | "
        + " | ".join(f"{c:.0f} bohr" for c in sorted({k[0] for k in groups}))
        + " |"
    )
    print("|---|---|" + "---|" * len({k[0] for k in groups}))
    by_name = defaultdict(dict)
    for r in rows:
        by_name[(r["name"], r["atoms"])][r["cutoff3_bohr"]] = r["triangles"]
    for (name, atoms), counts in sorted(by_name.items()):
        print(
            f"| {name} | {atoms} | " + " | ".join(f"{counts[c]:.2e}" for c in sorted(counts)) + " |"
        )


if __name__ == "__main__":
    main()
