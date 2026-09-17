#!/usr/bin/env python3
"""Hybrid placements, process-grid shapes and memory, from the campaign.

    ./scripts/plot_paper_matrix.py [all.csv] [figures/]

Reads the campaign matrix (all.csv in the root) and writes two figures of
the report: hybrid.pdf, every processes x threads pair at 224^3 for both
backends (phase 12), and shapes.pdf, the fastest and the slowest process
grid for each number of processes (phase 11).  The memory table (phase 14)
has no figure: its numbers are printed with the others, for the text.
All three use SIMD; the pipeline runs with the fixed batch of 64 lines the
campaign used.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedLocator, NullLocator

import paperfig

GRID = ("224", "224", "224")
BACKENDS = [("schur", "Schur"), ("pipeline", "pipeline, $B=64$")]
# The shapes of phase 11 were all measured on this node from two processes
# up; the few rows elsewhere repeat a shape and would mix two machines.
SHAPE_NODE = "cpu05"
# What MPI_Dims_create returns for these counts in three dimensions.
DEFAULT_SHAPE = {1: "1x1x1", 2: "2x1x1", 4: "2x2x1", 7: "7x1x1",
                 8: "2x2x2", 14: "7x2x1", 28: "7x2x2", 56: "7x4x2"}
# Log axes with plain numbers: powers of ten alone leave too few labels.
TIME_TICKS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
# The sequential blue of the ParaView stills, light (fast) to dark (slow).
HEAT = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b"]


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("matrix", nargs="?", type=Path,
                        default=root / "all.csv")
    parser.add_argument("figures", nargs="?", type=Path,
                        default=root / "report" / "figures")
    return parser.parse_args()


def load(path):
    with path.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["status"] == "ok"
                and r["simd"] == "1" and (r["nx"], r["ny"], r["nz"]) == GRID]
    if not rows:
        raise SystemExit(f"no 224^3 SIMD rows in {path}")
    return rows


def shape(row):
    return f'{row["px"]}x{row["py"]}x{row["pz"]}'


def plain_log_axis(axis, low, high):
    ticks = [t for t in TIME_TICKS if low <= t <= high]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_ticklabels([str(t) for t in ticks])


def fastest(rows):
    """Best wall time of each (ranks, threads) pair, or of each shape."""
    best = defaultdict(lambda: math.inf)
    for row in rows:
        key = row["_key"]
        best[key] = min(best[key], float(row["wall_ms"]))
    return best


def hybrid(rows, out, preview):
    # The binary with OpenMP in every cell, so the corner 1x1 is the same
    # program as the rest of its row, not the serial one.
    rows = [dict(r, _key=(int(r["ranks"]), int(r["threads"]))) for r in rows
            if r["phase"].startswith("12") and r["omp"] == "1"]
    tables = {backend: fastest([r for r in rows if r["backend"] == backend])
              for backend, _ in BACKENDS}
    ranks = sorted({k[0] for t in tables.values() for k in t})
    threads = sorted({k[1] for t in tables.values() for k in t})
    values = [v for t in tables.values() for v in t.values()]
    norm = LogNorm(vmin=min(values), vmax=max(values))
    cmap = LinearSegmentedColormap.from_list("heat", HEAT)
    cmap.set_bad("white")

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True,
                             constrained_layout=True)
    print("hybrid, 224^3, SIMD, ms per step (rows processes, columns threads)")
    for ax, (backend, label) in zip(axes, BACKENDS):
        table = tables[backend]
        grid = [[table.get((p, t), math.nan) for t in threads] for p in ranks]
        image = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
        print(f"  {backend}: " + " ".join(f"{t:>6}" for t in threads))
        for i, p in enumerate(ranks):
            print(f"  {p:>{len(backend) + 1}}  " + " ".join(
                f"{grid[i][j]:6.0f}" if not math.isnan(grid[i][j])
                else "     ." for j in range(len(threads))))
            for j, value in enumerate(grid[i]):
                if math.isnan(value):
                    continue
                dark = norm(value) > 0.55
                ax.text(j, i, f"{value:.0f}", ha="center", va="center",
                        fontsize=5.6,
                        color="white" if dark else paperfig.INK)
        best = min(table, key=table.get)
        print(f"  best {best[0]}x{best[1]}: {table[best]:.1f} ms")
        ax.set_xticks(range(len(threads)))
        ax.set_xticklabels([str(t) for t in threads])
        ax.set_yticks(range(len(ranks)))
        ax.set_yticklabels([str(p) for p in ranks])
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(label, loc="left")
        ax.set_xlabel("threads per process")
    axes[0].set_ylabel("processes")
    bar = fig.colorbar(image, ax=axes, shrink=0.9, pad=0.02)
    bar.set_label("ms per time step")
    bar.outline.set_visible(False)
    plain_log_axis(bar.ax.yaxis, norm.vmin, norm.vmax)
    bar.ax.tick_params(length=2, color=paperfig.AXIS)
    paperfig.save(fig, out, preview)


def shapes(rows, out, preview):
    rows = [dict(r, _key=(int(r["ranks"]), shape(r))) for r in rows
            if r["phase"].startswith("11") and r["omp"] == "0"
            and r["mpi"] == "1" and r["threads"] == "1"
            and r["node"] == SHAPE_NODE]

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7), sharey=True,
                             constrained_layout=True)
    print(f"shapes, 224^3, SIMD, one thread per process, node {SHAPE_NODE}")
    for index, (ax, (backend, label)) in enumerate(zip(axes, BACKENDS)):
        table = fastest([r for r in rows if r["backend"] == backend])
        ranks = sorted({p for p, _ in table})
        colour = paperfig.SERIES[index]
        best, worst, default = [], [], []
        for p in ranks:
            times = {s: v for (q, s), v in table.items() if q == p}
            b = min(times, key=times.get)
            w = max(times, key=times.get)
            d = times.get(DEFAULT_SHAPE[p])
            best.append(times[b])
            worst.append(times[w])
            default.append(d)
            ax.plot([p, p], [times[b], times[w]], color=paperfig.MUTED,
                    lw=0.8, zorder=2)
            print(f"  {backend} {p:3d} procs, {len(times):2d} shapes: best "
                  f"{b:>7} {times[b]:6.0f}, worst {w:>7} {times[w]:6.0f} "
                  f"(x{times[w] / times[b]:.2f}), MPI_Dims_create "
                  f"{DEFAULT_SHAPE[p]:>7} {d:6.0f} (+{100 * (d / times[b] - 1):.0f}%)")
            if p == ranks[-1]:
                for value, name in ((times[b], b), (times[w], w)):
                    ax.annotate(name.replace("x", r"$\times$"),
                                xy=(p, value), xytext=(5, 0),
                                textcoords="offset points", fontsize=7,
                                color=paperfig.INK_SECONDARY, va="center")
        ax.plot(ranks, best, color=colour, lw=1.3, marker="o", ms=4.2,
                mec="white", mew=0.7, label="fastest shape", zorder=3)
        ax.plot(ranks, worst, ls="none", marker="o", ms=4.2, mfc="white",
                mec=colour, mew=1.0, label="slowest shape", zorder=3)
        ax.plot(ranks, default, ls="none", marker="x", ms=4.0,
                color=paperfig.INK_SECONDARY, mew=0.9,
                label="MPI_Dims_create", zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([p for p in ranks if p != 7])
        ax.set_xticklabels([str(p) for p in ranks if p != 7])
        ax.minorticks_off()
        plain_log_axis(ax.yaxis, 100, 5000)
        ax.set_xlim(0.8, 160)
        paperfig.tidy(ax, label)
        ax.set_xlabel("processes")
    axes[0].set_ylabel("ms per time step")
    axes[0].legend(loc="lower left", handlelength=1.6)
    paperfig.save(fig, out, preview)


def memory(path):
    with path.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["status"] == "ok"
                and r["phase"].startswith("14") and r["rss_mb"]
                and r["simd"] == "1" and r["threads"] == "1"]
    peak = defaultdict(lambda: math.inf)
    for r in rows:
        key = (r["backend"], (r["nx"], r["ny"], r["nz"]), int(r["ranks"]),
               r["mpi"], r["omp"])
        peak[key] = min(peak[key], float(r["rss_mb"]))
    print("memory: peak RSS of the largest process, MiB, SIMD, one thread")
    for backend, _ in BACKENDS:
        small = peak[(backend, ("32", "32", "32"), 1, "1", "1")]
        serial = peak[(backend, ("32", "32", "32"), 1, "0", "0")]
        print(f"  {backend} 32^3: {small:.1f} with MPI and OpenMP, "
              f"{serial:.1f} serial")
        for p in (1, 2, 4, 7, 8, 14, 28, 56):
            value = peak.get((backend, GRID, p, "1", "1"))
            if value is None:
                continue
            print(f"    224^3 {p:3d} procs: {value:7.1f} per process, "
                  f"at most {p * value:7.0f} in all")


def main():
    args = parse_args()
    rows = load(args.matrix)
    paperfig.style()
    # Previews go to build/, which is not tracked: the figures folder holds
    # only what the paper includes.
    root = Path(__file__).resolve().parents[1]
    hybrid(rows, args.figures / "hybrid.pdf",
           root / "build" / "figures" / "hybrid.png")
    shapes(rows, args.figures / "shapes.pdf",
           root / "build" / "figures" / "shapes.png")
    memory(args.matrix)


if __name__ == "__main__":
    main()
