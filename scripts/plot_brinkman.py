#!/usr/bin/env python3
"""Figure of the Brinkman channel study: relative L2 error of u against dy.

    ./scripts/plot_brinkman.py [results.csv] [figure.pdf]

Reads docs/brinkman/results.csv (written by scripts/run_brinkman.sh) and
writes docs/brinkman/brinkman.pdf for the report, plus a PNG next to it.

One panel per case.  K is an ordered parameter, so the lines share one hue
from light (K = 1) to dark (smallest K), with a different marker each so the
figure still reads in grayscale.  The grey line gives the expected order:
2 for uniform K, 1 across the jump of the porous layer.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

import paperfig

PANELS = [
    ("uniform", "Uniform $K$", 2),
    ("layer", "Porous layer, free fluid above", 1),
]


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="?", type=Path,
                        default=root / "docs" / "brinkman" / "results.csv")
    parser.add_argument("figure", nargs="?", type=Path,
                        default=root / "docs" / "brinkman" / "brinkman.pdf")
    return parser.parse_args()


def series(rows, case):
    by_k = {}
    for row in rows:
        if row["case"] != case:
            continue
        by_k.setdefault(float(row["K"]), []).append(
            (float(row["dy"]), float(row["rel_l2"])))
    return {k: sorted(points) for k, points in by_k.items()}


def main():
    args = parse_args()
    with args.results.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no rows in {args.results}")

    paperfig.style()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7), sharey=True,
                             constrained_layout=True)

    for ax, (case, title, order) in zip(axes, PANELS):
        data = series(rows, case)
        ks = sorted(data, reverse=True)
        if len(ks) > len(paperfig.RAMP):
            raise SystemExit(f"{case}: {len(ks)} permeabilities, at most "
                             f"{len(paperfig.RAMP)} fit the ramp")

        for index, k in enumerate(ks):
            paperfig.line(ax, [p[0] for p in data[k]],
                          [p[1] for p in data[k]], index,
                          f"$K = 10^{{{round(math.log10(k))}}}$",
                          palette=paperfig.RAMP)
        ax.set_xscale("log")
        ax.set_yscale("log")

        # In the corner above the finest grids, where no series passes.
        fine = min(p[0] for points in data.values() for p in points)
        top = max(p[1] for points in data.values() for p in points)
        paperfig.slope_guide(ax, fine * 1.05, top * 0.25, order)

        paperfig.tidy(ax, title)
        ax.set_xlabel(r"$\Delta y$")

    axes[0].set_ylabel(r"relative $L^2$ error of $\mathbf{u}$")
    # The empty corner of the uniform panel: coarse grids, small errors.
    axes[0].legend(loc="lower right", handlelength=1.8)

    paperfig.save(fig, args.figure)


if __name__ == "__main__":
    main()
