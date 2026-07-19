#!/usr/bin/env python3
"""Generate the end-to-end solver performance charts from median samples."""

from __future__ import annotations

import argparse
import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path


WIDTH = 1000
HEIGHT = 560
LEFT = 105
RIGHT = 965
TOP = 105
BOTTOM = 465
COLORS = {"standard": "#1f77b4", "optimized": "#2ca02c"}
MARKERS = {"standard": "circle", "optimized": "square"}


@dataclass(frozen=True)
class Sample:
    backend: str
    extent: int
    timestep_ns_per_cell: float

    @property
    def mlups(self) -> float:
        return 1000.0 / self.timestep_ns_per_cell

    @property
    def seconds_per_cell(self) -> float:
        return self.timestep_ns_per_cell * 1.0e-9


def read_samples(path: Path) -> list[Sample]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    samples = [
        Sample(
            backend=row["backend"],
            extent=int(row["extent"]),
            timestep_ns_per_cell=float(row["timestep_ns_per_cell"]),
        )
        for row in rows
    ]
    if not samples:
        raise ValueError(f"no samples in {path}")
    backends = {sample.backend for sample in samples}
    if backends != set(COLORS):
        raise ValueError(f"expected {sorted(COLORS)}, found {sorted(backends)}")
    extents = {
        backend: [
            sample.extent for sample in samples if sample.backend == backend
        ]
        for backend in backends
    }
    if len({tuple(sorted(values)) for values in extents.values()}) != 1:
        raise ValueError("backends must contain the same grid extents")
    return samples


def nice_ticks(values: list[float], count: int = 5) -> list[float]:
    low = min(values)
    high = max(values)
    padding = max((high - low) * 0.15, abs(high) * 0.03)
    low -= padding
    high += padding
    raw_step = (high - low) / count
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    fraction = raw_step / magnitude
    nice_fraction = next(value for value in (1.0, 2.0, 5.0, 10.0)
                         if value >= fraction)
    step = nice_fraction * magnitude
    first = math.floor(low / step) * step
    last = math.ceil(high / step) * step
    ticks = []
    value = first
    while value <= last + step * 0.5:
        ticks.append(value)
        value += step
    return ticks


def marker(kind: str, x: float, y: float, color: str) -> str:
    if kind == "square":
        return (
            f'<rect x="{x - 6:.2f}" y="{y - 6:.2f}" width="12" height="12" '
            f'rx="1" fill="{color}"/>'
        )
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}"/>'


def write_chart(samples: list[Sample], output: Path, metric: str) -> None:
    if metric == "mlups":
        values = {sample: sample.mlups for sample in samples}
        ylabel = "Performance (MLUPS)"
        title = "Solver throughput — Single Core · Apple M1"
        tick_scale = 1.0
        tick_suffix = ""
        tick_digits = 1
    else:
        values = {sample: sample.seconds_per_cell for sample in samples}
        ylabel = "Time per cell (seconds)"
        title = "Solver time per cell — Single Core · Apple M1"
        tick_scale = 1.0e7
        tick_suffix = "×10⁻⁷"
        tick_digits = 2

    extents = sorted({sample.extent for sample in samples})
    y_ticks = nice_ticks(list(values.values()))
    y_min = y_ticks[0]
    y_max = y_ticks[-1]

    def x_position(extent: int) -> float:
        return LEFT + (extent - extents[0]) / (extents[-1] - extents[0]) * (
            RIGHT - LEFT
        )

    def y_position(value: float) -> float:
        return BOTTOM - (value - y_min) / (y_max - y_min) * (BOTTOM - TOP)

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-label="{html.escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#2b2b2b}'
        '.title{font-size:22px;font-weight:600}.subtitle{font-size:14px;fill:#555}'
        '.axis{font-size:15px;font-weight:600}.tick{font-size:13px;fill:#444}'
        '.legend{font-size:14px}.grid{stroke:#d9d9d9;stroke-width:1;opacity:.65}'
        '.frame{stroke:#777;stroke-width:1;fill:none}.series{fill:none;stroke-width:3}'
        '</style>',
        f'<text class="title" x="{WIDTH / 2}" y="34" text-anchor="middle">'
        f'{html.escape(title)}</text>',
        '<text class="subtitle" x="500" y="60" text-anchor="middle">'
        'Release · double precision · paper workload · median of 5 runs</text>',
    ]

    for tick in y_ticks:
        y = y_position(tick)
        svg.append(
            f'<line class="grid" x1="{LEFT}" y1="{y:.2f}" '
            f'x2="{RIGHT}" y2="{y:.2f}"/>'
        )
        label = f"{tick * tick_scale:.{tick_digits}f}"
        svg.append(
            f'<text class="tick" x="{LEFT - 14}" y="{y + 5:.2f}" '
            f'text-anchor="end">{label}</text>'
        )
    for extent in extents:
        x = x_position(extent)
        svg.append(
            f'<line class="grid" x1="{x:.2f}" y1="{TOP}" '
            f'x2="{x:.2f}" y2="{BOTTOM}"/>'
        )
        svg.append(
            f'<text class="tick" x="{x:.2f}" y="{BOTTOM + 25}" '
            f'text-anchor="middle">{extent}</text>'
        )

    svg.extend([
        f'<rect class="frame" x="{LEFT}" y="{TOP}" '
        f'width="{RIGHT - LEFT}" height="{BOTTOM - TOP}"/>',
        f'<text class="axis" x="{(LEFT + RIGHT) / 2}" y="{HEIGHT - 30}" '
        'text-anchor="middle">Grid dimension (N for N³ grid)</text>',
        f'<text class="axis" x="28" y="{(TOP + BOTTOM) / 2}" '
        f'text-anchor="middle" transform="rotate(-90 28 {(TOP + BOTTOM) / 2})">'
        f'{html.escape(ylabel)}</text>',
    ])
    if tick_suffix:
        svg.append(
            f'<text class="tick" x="{LEFT}" y="{TOP - 10}">{tick_suffix}</text>'
        )

    for backend in ("standard", "optimized"):
        backend_samples = sorted(
            (sample for sample in samples if sample.backend == backend),
            key=lambda sample: sample.extent,
        )
        points = [
            (x_position(sample.extent), y_position(values[sample]))
            for sample in backend_samples
        ]
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        color = COLORS[backend]
        svg.append(
            f'<polyline class="series" stroke="{color}" points="{point_text}"/>'
        )
        svg.extend(marker(MARKERS[backend], x, y, color) for x, y in points)

    legend_x = RIGHT - 135
    for offset, backend in enumerate(("standard", "optimized")):
        y = TOP + 24 + offset * 26
        color = COLORS[backend]
        svg.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        svg.append(marker(MARKERS[backend], legend_x + 14, y, color))
        svg.append(
            f'<text class="legend" x="{legend_x + 38}" y="{y + 5}">'
            f'{backend.upper()}</text>'
        )

    svg.append("</svg>")
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MLUPS and seconds-per-cell SVG charts."
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("docs/solver-performance.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()

    samples = read_samples(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_chart(samples, args.output_dir / "solver-performance-mlups.svg", "mlups")
    write_chart(
        samples,
        args.output_dir / "solver-performance-time-per-cell.svg",
        "seconds_per_cell",
    )


if __name__ == "__main__":
    main()
