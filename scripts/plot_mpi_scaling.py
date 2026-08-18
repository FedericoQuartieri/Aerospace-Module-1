#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict
from html import escape
from pathlib import Path


COLORS = ["#2563eb", "#d97706", "#059669", "#7c3aed"]


def latest_results(build_dir):
    candidates = list(build_dir.glob("mpi_scaling_*.csv"))
    return max(candidates, key=lambda path: path.stat().st_mtime) \
        if candidates else build_dir / "mpi_scaling.csv"


def arguments():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot MPI strong scaling and scalar/SIMD execution times."
    )
    parser.add_argument("input", nargs="?", type=Path,
                        default=latest_results(root / "build"))
    parser.add_argument("output_base", nargs="?", type=Path,
                        help="output base name, without _strong/_comparison")
    parser.add_argument("--comparison-rank", type=int, default=4,
                        help="MPI rank count used in the bar chart (default: 4)")
    return parser.parse_args()


def load(path):
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            raw_rows = list(csv.DictReader(stream))
    except FileNotFoundError:
        raise SystemExit(f"Results not found: {path}") from None

    required = {"ranks", "time_s"}
    if not raw_rows or not required.issubset(raw_rows[0]):
        raise SystemExit(f"Invalid or empty scaling CSV: {path}")

    rows = []
    for row in raw_rows:
        parsed = {
            "grid": int(row.get("grid") or 128),
            "simd": int(row.get("simd") or 0),
            "ranks": int(row["ranks"]),
            "time_s": float(row["time_s"]),
        }
        if parsed["time_s"] <= 0 or not math.isfinite(parsed["time_s"]):
            raise SystemExit("The scaling CSV contains invalid times")
        rows.append(parsed)
    return rows


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#172033}'
        '.axis{stroke:#64748b;stroke-width:1}.grid{stroke:#e2e8f0}'
        '.title{font-size:18px;font-weight:600}.tick{font-size:11px}'
        '.label,.legend{font-size:12px}.value{font-size:11px;font-weight:600}'
        '</style>',
    ]


def strong_scaling_svg(rows):
    series = defaultdict(list)
    for row in rows:
        series[(row["grid"], row["simd"])].append(row)
    for points in series.values():
        points.sort(key=lambda point: point["ranks"])
    series = dict(sorted(series.items()))
    if not series:
        raise SystemExit("No MPI data found in scaling CSV")

    ranks = sorted({point["ranks"] for points in series.values()
                    for point in points})
    if len(ranks) < 2:
        raise SystemExit("Strong scaling plot requires at least two rank counts")
    min_rank, max_rank = min(ranks), max(ranks)
    width, height = 920, 475
    svg = svg_header(width, height)
    svg.append('<text class="title" x="460" y="30" text-anchor="middle">'
               'MPI strong scaling</text>')

    legend_x = 35
    for index, ((grid, simd), _) in enumerate(series.items()):
        color = COLORS[index % len(COLORS)]
        label = f'N={grid}³ — {"SIMD" if simd else "scalar"}'
        svg += [f'<line x1="{legend_x}" y1="57" x2="{legend_x + 25}" '
                f'y2="57" stroke="{color}" stroke-width="3"/>',
                f'<text class="legend" x="{legend_x + 31}" y="61">'
                f'{escape(label)}</text>']
        legend_x += 205
    svg += ['<line x1="700" y1="57" x2="730" y2="57" stroke="#64748b" '
            'stroke-width="2" stroke-dasharray="6 4"/>',
            '<text class="legend" x="737" y="61">ideal</text>']

    panels = [("Execution time", "time_s", "seconds"),
              ("Speedup", "speedup", "T₁ / Tₚ")]
    for panel_index, (title, key, ylabel) in enumerate(panels):
        left = 65 + panel_index * 445
        top, plot_width, plot_height = 110, 370, 280
        bottom = top + plot_height

        values = []
        plotted = {}
        for series_key, points in series.items():
            baseline = next((point["time_s"] for point in points
                             if point["ranks"] == 1), points[0]["time_s"])
            plotted[series_key] = []
            for point in points:
                value = point["time_s"] if key == "time_s" \
                    else baseline / point["time_s"]
                plotted[series_key].append((point["ranks"], value, baseline))
                values.append(value)

        ymax = max(values) * 1.12
        if key == "speedup":
            ymax = max(ymax, max_rank * 1.08)
        xcoord = lambda value: left + plot_width * \
            (value - min_rank) / (max_rank - min_rank)
        ycoord = lambda value: bottom - plot_height * value / ymax

        svg.append(f'<text class="title" x="{left + plot_width / 2}" '
                   f'y="92" text-anchor="middle">{title}</text>')
        for tick in range(6):
            value = ymax * tick / 5
            y = ycoord(value)
            svg += [f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                    f'x2="{left + plot_width}" y2="{y:.2f}"/>',
                    f'<text class="tick" x="{left - 8}" y="{y + 4:.2f}" '
                    f'text-anchor="end">{value:.3g}</text>']
        for rank in ranks:
            x = xcoord(rank)
            svg += [f'<line class="axis" x1="{x:.2f}" y1="{bottom}" '
                    f'x2="{x:.2f}" y2="{bottom + 5}"/>',
                    f'<text class="tick" x="{x:.2f}" y="{bottom + 20}" '
                    f'text-anchor="middle">{rank}</text>']
        svg += [f'<line class="axis" x1="{left}" y1="{top}" '
                f'x2="{left}" y2="{bottom}"/>',
                f'<line class="axis" x1="{left}" y1="{bottom}" '
                f'x2="{left + plot_width}" y2="{bottom}"/>',
                f'<text class="label" x="{left + plot_width / 2}" y="440" '
                f'text-anchor="middle">MPI ranks</text>',
                f'<text class="label" transform="translate({left - 47} '
                f'{top + plot_height / 2}) rotate(-90)" text-anchor="middle">'
                f'{ylabel}</text>']

        if key == "speedup":
            svg.append(f'<line x1="{xcoord(min_rank):.2f}" '
                       f'y1="{ycoord(min_rank):.2f}" '
                       f'x2="{xcoord(max_rank):.2f}" y2="{ycoord(max_rank):.2f}" '
                       f'stroke="#64748b" stroke-width="2" '
                       f'stroke-dasharray="6 4"/>')

        for index, points in enumerate(plotted.values()):
            color = COLORS[index % len(COLORS)]
            if key == "time_s":
                baseline = points[0][2]
                ideal = " ".join(
                    f'{xcoord(rank):.2f},{ycoord(baseline / rank):.2f}'
                    for rank in ranks)
                svg.append(f'<polyline points="{ideal}" fill="none" '
                           f'stroke="{color}" stroke-width="1.8" opacity="0.65" '
                           f'stroke-dasharray="6 4"/>')
            coordinates = " ".join(
                f'{xcoord(rank):.2f},{ycoord(value):.2f}'
                for rank, value, _ in points)
            svg.append(f'<polyline points="{coordinates}" fill="none" '
                       f'stroke="{color}" stroke-width="2.8"/>')
            for rank, value, _ in points:
                svg.append(f'<circle cx="{xcoord(rank):.2f}" '
                           f'cy="{ycoord(value):.2f}" r="4" fill="{color}"/>')

    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def comparison_svg(rows, comparison_rank):
    grids = sorted({row["grid"] for row in rows})
    configurations = [
        (0, 1, "MPI 1 rank", "scalar", "#93c5fd"),
        (0, comparison_rank, f"MPI {comparison_rank} ranks", "scalar", "#2563eb"),
        (1, 1, "MPI 1 rank", "SIMD", "#fbbf24"),
        (1, comparison_rank, f"MPI {comparison_rank} ranks", "SIMD", "#d97706"),
    ]
    values = {(row["grid"], row["simd"], row["ranks"]):
              row["time_s"] for row in rows}
    groups = []
    for grid in grids:
        bars = []
        for simd, ranks, line1, line2, color in configurations:
            value = values.get((grid, simd, ranks))
            if value is not None:
                bars.append((line1, line2, value, color))
        if bars:
            groups.append((grid, bars))
    if not groups:
        raise SystemExit("No execution-time configurations available")

    width = max(900, 110 + sum(len(bars) * 100 + 55 for _, bars in groups))
    height = 500
    left, top, bottom = 75, 70, 390
    plot_height = bottom - top
    ymax = max(value for _, bars in groups for _, _, value, _ in bars) * 1.15
    ycoord = lambda value: bottom - plot_height * value / ymax
    svg = svg_header(width, height)
    svg.append('<text class="title" x="50%" y="30" text-anchor="middle">'
               'MPI execution time: scalar and SIMD</text>')

    plot_right = width - 30
    for tick in range(6):
        value = ymax * tick / 5
        y = ycoord(value)
        svg += [f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                f'x2="{plot_right}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{left - 8}" y="{y + 4:.2f}" '
                f'text-anchor="end">{value:.3g}</text>']
    svg += [f'<line class="axis" x1="{left}" y1="{top}" '
            f'x2="{left}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{bottom}" '
            f'x2="{plot_right}" y2="{bottom}"/>',
            f'<text class="label" transform="translate(22 '
            f'{top + plot_height / 2}) rotate(-90)" text-anchor="middle">'
            'execution time [s]</text>']

    x = left + 35
    for grid, bars in groups:
        group_start = x
        for line1, line2, value, color in bars:
            bar_width = 62
            y = ycoord(value)
            svg += [f'<rect x="{x}" y="{y:.2f}" width="{bar_width}" '
                    f'height="{bottom - y:.2f}" fill="{color}"/>',
                    f'<text class="value" x="{x + bar_width / 2}" '
                    f'y="{y - 7:.2f}" text-anchor="middle">{value:.2f}</text>',
                    f'<text class="tick" x="{x + bar_width / 2}" y="{bottom + 18}" '
                    f'text-anchor="middle">{escape(line1)}</text>',
                    f'<text class="tick" x="{x + bar_width / 2}" y="{bottom + 33}" '
                    f'text-anchor="middle">{escape(line2)}</text>']
            x += 100
        group_center = (group_start + x - 38) / 2
        svg.append(f'<text class="label" x="{group_center:.2f}" y="465" '
                   f'text-anchor="middle">N={grid}³</text>')
        x += 55

    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def main():
    args = arguments()
    rows = load(args.input)
    base = args.output_base or args.input.with_suffix("")
    if base.suffix == ".svg":
        base = base.with_suffix("")
    strong_output = base.with_name(base.name + "_strong.svg")
    comparison_output = base.with_name(base.name + "_comparison.svg")
    strong_output.parent.mkdir(parents=True, exist_ok=True)
    strong_output.write_text(strong_scaling_svg(rows), encoding="utf-8")
    comparison_output.write_text(
        comparison_svg(rows, args.comparison_rank), encoding="utf-8"
    )
    print(f"Strong-scaling plot written to {strong_output}")
    print(f"Execution-time comparison written to {comparison_output}")


if __name__ == "__main__":
    main()
