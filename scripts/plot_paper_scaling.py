#!/usr/bin/env python3
"""Strong and weak scaling of the two backends, from the scaling campaign.

    ./scripts/plot_paper_scaling.py [all.csv] [figure.pdf]

Reads the campaign matrix (all.csv in the root, phase 14) and writes
report/figures/scaling.pdf.  Strong scaling: 224^3 with SIMD, one thread
per process, from 1 to 56 processes on one node.  Weak scaling: 64^3 cells
per process on the same placements.  Every point is the best of the
repetitions on the node the campaign fixed, and the numbers behind the
figure are printed for the text.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

import paperfig

STRONG_GRID = ("224", "224", "224")
WEAK = [(1, ("64", "64", "64")), (2, ("128", "64", "64")),
        (4, ("128", "128", "64")), (8, ("128", "128", "128")),
        (28, ("224", "224", "128")), (56, ("224", "224", "224"))]
BACKENDS = [("schur", "Schur"), ("pipeline", "pipeline")]


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("matrix", nargs="?", type=Path,
                        default=root / "all.csv")
    parser.add_argument("figure", nargs="?", type=Path,
                        default=root / "report" / "figures" / "scaling.pdf")
    return parser.parse_args()


def load(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [r for r in rows if r["phase"].startswith("14")
            and r["status"] == "ok" and r["threads"] == "1"
            and r["simd"] == "1"]
    if not rows:
        raise SystemExit(f"no phase-14 SIMD rows in {path}")
    return rows


def best(rows, backend, ranks, grid, omp):
    """Best wall time per step, its MPI share and the process grid."""
    found = [r for r in rows if r["backend"] == backend
             and int(r["ranks"]) == ranks and r["omp"] == omp
             and (r["nx"], r["ny"], r["nz"]) == grid]
    if not found:
        return None
    row = min(found, key=lambda r: float(r["wall_ms"]))
    return (float(row["wall_ms"]), float(row["mpi_ms"]),
            f'{row["px"]}x{row["py"]}x{row["pz"]}', row["node"])


def main():
    args = parse_args()
    rows = load(args.matrix)
    ranks = sorted({int(r["ranks"]) for r in rows
                    if (r["nx"], r["ny"], r["nz"]) == STRONG_GRID
                    and r["omp"] == "1"})

    paperfig.style()
    fig, (strong, weak) = plt.subplots(1, 2, figsize=(6.3, 2.7),
                                       constrained_layout=True)

    print("strong scaling, 224^3, SIMD, one thread per process")
    for index, (backend, label) in enumerate(BACKENDS):
        serial = best(rows, backend, 1, STRONG_GRID, "0")
        points = [(p, best(rows, backend, p, STRONG_GRID, "1"))
                  for p in ranks]
        points = [(p, b) for p, b in points if b]
        x = [p for p, _ in points]
        y = [b[0] for _, b in points]
        paperfig.line(strong, x, y, index, label)
        print(f"  {backend}: serial {serial[0]:.1f} ms on {serial[3]}")
        for p, b in points:
            print(f"    {p:3d} procs {b[2]:>6}: {b[0]:7.1f} ms, MPI "
                  f"{100 * b[1] / b[0]:4.1f}%, speedup {serial[0] / b[0]:5.2f}"
                  f" vs serial, {y[0] / b[0]:5.2f} vs 1 proc, "
                  f"efficiency {100 * y[0] / b[0] / p:4.1f}%")
        if index == 0:
            ideal = [y[0] / p for p in x]
            strong.plot(x, ideal, color=paperfig.MUTED, lw=0.8, zorder=2)
            strong.annotate("ideal", xy=(x[-1], ideal[-1]), xytext=(4, 0),
                            textcoords="offset points", color=paperfig.MUTED,
                            fontsize=8, va="center")

    strong.set_xscale("log", base=2)
    strong.set_yscale("log")
    # 7 and 8 sit too close on the axis for two labels: 7 keeps its point
    # and the text says what it is.
    strong.set_xticks([p for p in ranks if p != 7])
    strong.set_xticklabels([str(p) for p in ranks if p != 7])
    strong.minorticks_off()
    paperfig.tidy(strong, "Strong scaling, $224^3$")
    strong.set_xlabel("processes")
    strong.set_ylabel("ms per time step")
    strong.legend(loc="upper right", handlelength=1.8)

    print("weak scaling, 64^3 cells per process")
    for index, (backend, label) in enumerate(BACKENDS):
        points = [(p, best(rows, backend, p, grid, "1")) for p, grid in WEAK]
        points = [(p, b) for p, b in points if b]
        x = [p for p, _ in points]
        y = [b[0] for _, b in points]
        paperfig.line(weak, x, y, index, label)
        for (p, b), grid in zip(points, [g for _, g in WEAK]):
            cells = int(grid[0]) * int(grid[1]) * int(grid[2]) / p
            print(f"  {backend} {p:3d} procs {'x'.join(grid):>11}: "
                  f"{b[0]:6.1f} ms, MPI {100 * b[1] / b[0]:4.1f}%, "
                  f"{cells / 1e3:5.0f}k cells/proc, "
                  f"time x{b[0] / y[0]:4.2f}")

    weak.set_xscale("log", base=2)
    weak.set_xticks([p for p, _ in WEAK])
    weak.set_xticklabels([str(p) for p, _ in WEAK])
    weak.minorticks_off()
    weak.set_ylim(bottom=0)
    paperfig.tidy(weak, "Weak scaling, $64^3$ cells per process")
    weak.set_xlabel("processes")
    weak.set_ylabel("ms per time step")

    # The preview goes to build/, which is not tracked: report/figures/
    # holds only what the paper includes.
    root = Path(__file__).resolve().parents[1]
    paperfig.save(fig, args.figure,
                  root / "build" / "figures" / f"{args.figure.stem}.png")


if __name__ == "__main__":
    main()
