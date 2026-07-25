#!/usr/bin/env python3

import csv
import math
import sys
from html import escape
from pathlib import Path


COLORS = {
    "L2_ux": "#2563eb",
    "L2_uy": "#dc2626",
    "L2_uz": "#16a34a",
    "L2_p": "#7c3aed",
}


def load_results(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def number(value):
    return float(value)


def observed_rate(rows, error_key, scale_key, reference_power):
    fine, coarse = rows[0], rows[1]

    return math.log(number(coarse[error_key]) / number(fine[error_key])) / abs(
        math.log(number(coarse[scale_key]) / number(fine[scale_key]))
    )


def log_ticks(minimum, maximum):
    low = math.floor(math.log10(minimum))
    high = math.ceil(math.log10(maximum))
    return [10.0**exponent for exponent in range(low, high + 1)]


def svg_text(x, y, text, size=14, anchor="middle", extra=""):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'text-anchor="{anchor}" {extra}>{escape(text)}</text>'
    )


def draw_panel(
    rows,
    series,
    title,
    scale_key,
    scale_label,
    reference_power,
    x0,
    y0,
    width,
    height,
):
    rows = sorted(rows, key=lambda row: number(row[scale_key]))
    x_values = [number(row[scale_key]) for row in rows]
    data = {
        key: [number(row[key]) for row in rows]
        for key, _ in series
    }

    anchor_x = min(x_values) if reference_power < 0 else max(x_values)
    anchor_index = x_values.index(anchor_x)
    reference_anchor = max(values[anchor_index] for values in data.values()) * 1.6
    reference = [
        reference_anchor * (x_value / anchor_x) ** reference_power
        for x_value in x_values
    ]

    all_y = [value for values in data.values() for value in values] + reference
    log_x_min = math.log10(min(x_values))
    log_x_max = math.log10(max(x_values))
    log_y_min = math.log10(min(all_y))
    log_y_max = math.log10(max(all_y))
    x_padding = max(0.08 * (log_x_max - log_x_min), 0.04)
    y_padding = max(0.10 * (log_y_max - log_y_min), 0.08)
    log_x_min -= x_padding
    log_x_max += x_padding
    log_y_min -= y_padding
    log_y_max += y_padding

    def map_x(value):
        return x0 + (math.log10(value) - log_x_min) / (log_x_max - log_x_min) * width

    def map_y(value):
        return y0 + height - (
            (math.log10(value) - log_y_min) / (log_y_max - log_y_min) * height
        )

    elements = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" '
        'fill="#ffffff" stroke="#cbd5e1"/>',
        svg_text(x0 + width / 2, y0 - 22, title, size=18),
    ]

    for tick in log_ticks(10**log_y_min, 10**log_y_max):
        if 10**log_y_min <= tick <= 10**log_y_max:
            y = map_y(tick)
            elements.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + width}" y2="{y:.1f}" '
                'stroke="#e2e8f0"/>'
            )
            elements.append(
                svg_text(x0 - 10, y + 5, f"{tick:.0e}", size=12, anchor="end")
            )

    for value in x_values:
        x = map_x(value)
        elements.append(
            f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + height}" '
            'stroke="#f1f5f9"/>'
        )
        elements.append(
            svg_text(x, y0 + height + 22, f"{value:.3g}", size=12)
        )

    for key, label in series:
        points = " ".join(
            f"{map_x(x_value):.1f},{map_y(y_value):.1f}"
            for x_value, y_value in zip(x_values, data[key])
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{COLORS[key]}" '
            'stroke-width="2.5"/>'
        )
        for x_value, y_value in zip(x_values, data[key]):
            elements.append(
                f'<circle cx="{map_x(x_value):.1f}" cy="{map_y(y_value):.1f}" '
                f'r="4" fill="{COLORS[key]}"/>'
            )

    reference_points = " ".join(
        f"{map_x(x_value):.1f},{map_y(y_value):.1f}"
        for x_value, y_value in zip(x_values, reference)
    )
    elements.append(
        f'<polyline points="{reference_points}" fill="none" stroke="#334155" '
        'stroke-width="2" stroke-dasharray="7 5"/>'
    )

    legend_x = x0 + 14
    legend_y = y0 + 22
    for index, (key, label) in enumerate(series):
        rate = observed_rate(rows, key, scale_key, reference_power)
        y = legend_y + index * 22
        elements.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" '
            f'stroke="{COLORS[key]}" stroke-width="3"/>'
        )
        elements.append(
            svg_text(
                legend_x + 31,
                y + 5,
                f"{label}, finest rate={rate:.2f}",
                size=12,
                anchor="start",
            )
        )

    reference_y = legend_y + len(series) * 22
    elements.append(
        f'<line x1="{legend_x}" y1="{reference_y}" '
        f'x2="{legend_x + 24}" y2="{reference_y}" '
        'stroke="#334155" stroke-width="2" stroke-dasharray="7 5"/>'
    )
    elements.append(
        svg_text(legend_x + 31, reference_y + 5, "second-order reference", 12, "start")
    )

    elements.append(svg_text(x0 + width / 2, y0 + height + 48, scale_label, size=14))
    elements.append(
        svg_text(
            x0 - 58,
            y0 + height / 2,
            "L2 error",
            size=14,
            extra=f'transform="rotate(-90 {x0 - 58:.1f} {y0 + height / 2:.1f})"',
        )
    )
    return elements


def write_figure(rows, output_path, title, series):
    spatial = [row for row in rows if row["study"] == "spatial"]
    temporal = [row for row in rows if row["study"] == "temporal"]

    width = 1200
    height = 560
    panel_width = 470
    panel_height = 350
    panel_y = 105
    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#0f172a">',
        svg_text(width / 2, 38, title, size=24),
    ]
    elements.extend(
        draw_panel(
            spatial,
            series,
            "Spatial convergence",
            "h",
            "Grid spacing h",
            2,
            105,
            panel_y,
            panel_width,
            panel_height,
        )
    )
    elements.extend(
        draw_panel(
            temporal,
            series,
            "Temporal convergence",
            "dt",
            "Time step dt",
            2,
            700,
            panel_y,
            panel_width,
            panel_height,
        )
    )
    elements.extend(["</g>", "</svg>"])
    output_path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_convergence.py RESULTS.csv OUTPUT_DIR")

    rows = load_results(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    write_figure(
        rows,
        output_dir / "convergence_velocity.svg",
        "Velocity convergence",
        [("L2_ux", "u_x"), ("L2_uy", "u_y"), ("L2_uz", "u_z")],
    )
    write_figure(
        rows,
        output_dir / "convergence_pressure.svg",
        "Pressure convergence (mean-aligned)",
        [("L2_p", "p")],
    )


if __name__ == "__main__":
    main()
