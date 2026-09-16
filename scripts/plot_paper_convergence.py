#!/usr/bin/env python3
"""Spatial and temporal convergence of the manufactured solution, one figure.

    ./scripts/plot_paper_convergence.py [results.csv] [figure.pdf]

Reads docs/convergence/results.csv, the cluster run of
scripts/run_convergence.sh, and writes report/figures/convergence.pdf.  Left
panel: L2 error against the grid spacing at fixed dt; right: against the
time step at fixed grid.  The three velocity components and the pressure are
the four series; the grey line is second order.  The observed orders between the two finest levels
are printed for the text.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

import paperfig

SERIES = [("L2_ux", "$u_x$"), ("L2_uy", "$u_y$"), ("L2_uz", "$u_z$"),
          ("L2_p", "$p$")]
PANELS = [("spatial", "h", "Spatial refinement", "$h$"),
          ("temporal", "dt", "Temporal refinement", r"$\Delta t$")]


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="?", type=Path,
                        default=root / "docs" / "convergence" / "results.csv")
    parser.add_argument("figure", nargs="?", type=Path,
                        default=root / "report" / "figures" / "convergence.pdf")
    return parser.parse_args()


def main():
    args = parse_args()
    with args.results.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no rows in {args.results}")

    paperfig.style()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7),
                             constrained_layout=True)

    for ax, (study, scale_key, title, xlabel) in zip(axes, PANELS):
        data = sorted((r for r in rows if r["study"] == study),
                      key=lambda r: float(r[scale_key]))
        if len(data) < 2:
            raise SystemExit(f"{study}: fewer than two levels")
        x = [float(r[scale_key]) for r in data]
        print(f"{study}: levels {[r['N'] if study == 'spatial' else r['dt'] for r in data]}")
        for index, (key, label) in enumerate(SERIES):
            y = [float(r[key]) for r in data]
            paperfig.line(ax, x, y, index, label)
            rate = math.log(y[1] / y[0]) / math.log(x[1] / x[0])
            print(f"  {key}: finest order {rate:.2f}, errors "
                  + ", ".join(f"{v:.3e}" for v in y))
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Below the lowest point: the series are at most second order, so
        # a steeper line that starts under them never crosses them.
        low = min(float(r[k]) for r in data for k, _ in SERIES)
        paperfig.slope_guide(ax, x[0] * 1.05, low * 0.45, 2, span=2.0)
        paperfig.tidy(ax, title)
        ax.set_xlabel(xlabel)
        # One tick per level, no minor ticks: the levels are the point.
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.3g}" for v in x])
        ax.minorticks_off()

    axes[0].set_ylabel("$L^2$ error")
    axes[1].legend(loc="lower right", handlelength=1.8, ncol=2)

    # The preview goes to build/, which is not tracked: report/figures/
    # holds only what the paper includes.
    root = Path(__file__).resolve().parents[1]
    paperfig.save(fig, args.figure,
                  root / "build" / "figures" / f"{args.figure.stem}.png")


if __name__ == "__main__":
    main()
