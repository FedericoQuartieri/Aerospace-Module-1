#!/usr/bin/env python3

import argparse
import csv
import math
from html import escape
from pathlib import Path


SERIES_STYLE = {
    "L2_ux": {"color": "#2563eb", "marker": "circle"},
    "L2_uy": {"color": "#e11d48", "marker": "square"},
    "L2_uz": {"color": "#059669", "marker": "diamond"},
    "L2_p": {"color": "#7c3aed", "marker": "circle"},
}

CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 620
PANEL_Y = 92
PANEL_WIDTH = 550
PANEL_HEIGHT = 480
PLOT_OFFSET_X = 74
PLOT_OFFSET_Y = 118
PLOT_WIDTH = 446
PLOT_HEIGHT = 280


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate the static convergence figures from a results CSV."
    )
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=root / "build" / "convergence" / "results.csv",
        help="input CSV (default: build/convergence/results.csv)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=root / "docs" / "convergence",
        help="figure directory (default: docs/convergence)",
    )
    return parser.parse_args()


def load_results(csv_path):
    try:
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
    except FileNotFoundError:
        raise SystemExit(
            f"results file not found: {csv_path}\n"
            "Run ./scripts/run_convergence.sh first."
        ) from None

    if not rows:
        raise SystemExit(f"results file is empty: {csv_path}")

    required = {
        "study",
        "h",
        "dt",
        "L2_ux",
        "L2_uy",
        "L2_uz",
        "L2_p",
    }
    missing = required.difference(rows[0])
    if missing:
        raise SystemExit(
            f"results file is missing columns: {', '.join(sorted(missing))}"
        )

    return rows


def number(value):
    return float(value)


def validate_rows(rows, series, scale_key, study):
    if len(rows) < 2:
        raise SystemExit(f"{study} study needs at least two discretizations")

    for row in rows:
        values = [number(row[scale_key])]
        values.extend(number(row[key]) for key, _ in series)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise SystemExit(
                f"{study} study contains non-positive or non-finite plot data"
            )


def observed_rate(rows, error_key, scale_key):
    fine, coarse = sorted(rows, key=lambda row: number(row[scale_key]))[:2]
    error_ratio = number(coarse[error_key]) / number(fine[error_key])
    scale_ratio = number(coarse[scale_key]) / number(fine[scale_key])
    return math.log(error_ratio) / math.log(scale_ratio)


def log_tick_exponents(log_minimum, log_maximum, maximum_ticks=6):
    low = math.ceil(log_minimum)
    high = math.floor(log_maximum)
    if low > high:
        return []

    count = high - low + 1
    step = max(1, math.ceil(count / maximum_ticks))
    return list(range(low, high + 1, step))


def superscript(exponent):
    translation = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(exponent).translate(translation)


def format_log_tick(exponent):
    return f"10{superscript(exponent)}"


def format_scale_tick(value):
    if value < 0.01 or value >= 1000:
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"{float(mantissa):g}×10{superscript(int(exponent))}"
    return f"{value:.4g}"


def svg_text(x, y, text, class_name, anchor="middle", extra=""):
    attributes = (
        f'x="{x:.1f}" y="{y:.1f}" class="{class_name}" '
        f'text-anchor="{anchor}"'
    )
    if extra:
        attributes += f" {extra}"
    return f"<text {attributes}>{escape(text)}</text>"


def marker_element(x, y, marker, color, size=4.5):
    common = f'fill="#ffffff" stroke="{color}" stroke-width="2.4"'
    if marker == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size}" {common}/>'
    if marker == "square":
        side = 2.0 * size
        return (
            f'<rect x="{x - size:.1f}" y="{y - size:.1f}" '
            f'width="{side:.1f}" height="{side:.1f}" rx="1" {common}/>'
        )
    if marker == "diamond":
        points = (
            f"{x:.1f},{y - size - 0.5:.1f} "
            f"{x + size + 0.5:.1f},{y:.1f} "
            f"{x:.1f},{y + size + 0.5:.1f} "
            f"{x - size - 0.5:.1f},{y:.1f}"
        )
        return f'<polygon points="{points}" {common}/>'
    raise ValueError(f"unknown marker: {marker}")


def draw_legend_item(x, y, label, color, marker):
    elements = [
        f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 25:.1f}" y2="{y:.1f}" '
        f'stroke="{color}" stroke-width="2.5"/>',
        marker_element(x + 12.5, y, marker, color, size=3.8),
        svg_text(x + 34, y + 4.5, label, "legend", anchor="start"),
    ]
    return elements


def draw_reference_legend_item(x, y, label):
    return [
        f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 25:.1f}" y2="{y:.1f}" '
        'class="reference"/>',
        svg_text(x + 34, y + 4.5, label, "legend muted", anchor="start"),
    ]


def draw_panel(
    rows,
    series,
    panel_title,
    panel_subtitle,
    scale_key,
    scale_label,
    reference_label,
    panel_x,
    panel_id,
):
    validate_rows(rows, series, scale_key, panel_title)
    rows = sorted(rows, key=lambda row: number(row[scale_key]))
    x_values = [number(row[scale_key]) for row in rows]
    data = {key: [number(row[key]) for row in rows] for key, _ in series}

    reference_power = 2
    anchor_x = max(x_values)
    anchor_index = x_values.index(anchor_x)
    reference_anchor = max(
        values[anchor_index] for values in data.values()
    ) * 1.45
    reference = [
        reference_anchor * (x_value / anchor_x) ** reference_power
        for x_value in x_values
    ]

    all_y = [value for values in data.values() for value in values] + reference
    log_x_min = math.log10(min(x_values))
    log_x_max = math.log10(max(x_values))
    log_y_min = math.log10(min(all_y))
    log_y_max = math.log10(max(all_y))
    x_padding = max(0.07 * (log_x_max - log_x_min), 0.035)
    y_padding = max(0.11 * (log_y_max - log_y_min), 0.10)
    log_x_min -= x_padding
    log_x_max += x_padding
    log_y_min -= y_padding
    log_y_max += y_padding

    plot_x = panel_x + PLOT_OFFSET_X
    plot_y = PANEL_Y + PLOT_OFFSET_Y

    def map_x(value):
        fraction = (math.log10(value) - log_x_min) / (log_x_max - log_x_min)
        return plot_x + fraction * PLOT_WIDTH

    def map_y(value):
        fraction = (math.log10(value) - log_y_min) / (log_y_max - log_y_min)
        return plot_y + PLOT_HEIGHT - fraction * PLOT_HEIGHT

    elements = [
        f'<rect x="{panel_x}" y="{PANEL_Y}" width="{PANEL_WIDTH}" '
        f'height="{PANEL_HEIGHT}" rx="18" class="card"/>',
        svg_text(
            panel_x + 28,
            PANEL_Y + 34,
            panel_title,
            "panel-title",
            anchor="start",
        ),
        svg_text(
            panel_x + 28,
            PANEL_Y + 57,
            panel_subtitle,
            "panel-subtitle",
            anchor="start",
        ),
    ]

    legend_positions = [
        (panel_x + 28, PANEL_Y + 84),
        (panel_x + 288, PANEL_Y + 84),
        (panel_x + 28, PANEL_Y + 106),
        (panel_x + 288, PANEL_Y + 106),
    ]
    legend_index = 0
    for key, label in series:
        rate = observed_rate(rows, key, scale_key)
        legend_x, legend_y = legend_positions[legend_index]
        elements.extend(
            draw_legend_item(
                legend_x,
                legend_y,
                f"{label}  ·  finest slope {rate:.2f}",
                SERIES_STYLE[key]["color"],
                SERIES_STYLE[key]["marker"],
            )
        )
        legend_index += 1

    legend_x, legend_y = legend_positions[legend_index]
    elements.extend(
        draw_reference_legend_item(legend_x, legend_y, reference_label)
    )

    elements.append(
        f'<rect x="{plot_x}" y="{plot_y}" width="{PLOT_WIDTH}" '
        f'height="{PLOT_HEIGHT}" rx="8" class="plot-background"/>'
    )

    for exponent in log_tick_exponents(log_y_min, log_y_max):
        value = 10.0**exponent
        y = map_y(value)
        elements.extend(
            [
                f'<line x1="{plot_x}" y1="{y:.1f}" '
                f'x2="{plot_x + PLOT_WIDTH}" y2="{y:.1f}" class="grid"/>',
                svg_text(
                    plot_x - 12,
                    y + 4,
                    format_log_tick(exponent),
                    "tick",
                    anchor="end",
                ),
            ]
        )

    for value in x_values:
        x = map_x(value)
        elements.extend(
            [
                f'<line x1="{x:.1f}" y1="{plot_y}" '
                f'x2="{x:.1f}" y2="{plot_y + PLOT_HEIGHT}" class="grid vertical"/>',
                svg_text(
                    x,
                    plot_y + PLOT_HEIGHT + 25,
                    format_scale_tick(value),
                    "tick",
                ),
            ]
        )

    elements.append(f'<g clip-path="url(#{panel_id}-clip)">')
    for key, _ in series:
        color = SERIES_STYLE[key]["color"]
        points = " ".join(
            f"{map_x(x_value):.1f},{map_y(y_value):.1f}"
            for x_value, y_value in zip(x_values, data[key])
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'class="series-line"/>'
        )
        for x_value, y_value in zip(x_values, data[key]):
            elements.append(
                marker_element(
                    map_x(x_value),
                    map_y(y_value),
                    SERIES_STYLE[key]["marker"],
                    color,
                )
            )

    reference_points = " ".join(
        f"{map_x(x_value):.1f},{map_y(y_value):.1f}"
        for x_value, y_value in zip(x_values, reference)
    )
    elements.extend(
        [
            f'<polyline points="{reference_points}" fill="none" class="reference"/>',
            "</g>",
            svg_text(
                plot_x + PLOT_WIDTH / 2,
                plot_y + PLOT_HEIGHT + 58,
                scale_label,
                "axis-label",
            ),
            svg_text(
                panel_x + 24,
                plot_y + PLOT_HEIGHT / 2,
                "L² error",
                "axis-label",
                extra=(
                    f'transform="rotate(-90 {panel_x + 24:.1f} '
                    f'{plot_y + PLOT_HEIGHT / 2:.1f})"'
                ),
            ),
        ]
    )
    return elements


def write_figure(rows, output_path, title, series):
    spatial = [row for row in rows if row["study"] == "spatial"]
    temporal = [row for row in rows if row["study"] == "temporal"]

    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_WIDTH}" '
        f'height="{CANVAS_HEIGHT}" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}" '
        'role="img">',
        f"<title>{escape(title)}</title>",
        "<defs>",
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">',
        '<feDropShadow dx="0" dy="5" stdDeviation="7" '
        'flood-color="#0f172a" flood-opacity="0.08"/>',
        "</filter>",
        f'<clipPath id="spatial-clip"><rect x="{40 + PLOT_OFFSET_X}" '
        f'y="{PANEL_Y + PLOT_OFFSET_Y}" width="{PLOT_WIDTH}" '
        f'height="{PLOT_HEIGHT}" rx="8"/></clipPath>',
        f'<clipPath id="temporal-clip"><rect x="{610 + PLOT_OFFSET_X}" '
        f'y="{PANEL_Y + PLOT_OFFSET_Y}" width="{PLOT_WIDTH}" '
        f'height="{PLOT_HEIGHT}" rx="8"/></clipPath>',
        "<style>",
        "text { font-family: Inter, ui-sans-serif, -apple-system, "
        "BlinkMacSystemFont, \"Segoe UI\", sans-serif; fill: #172033; }",
        ".canvas { fill: #f6f8fc; }",
        ".card { fill: #ffffff; stroke: #dfe5ef; filter: url(#shadow); }",
        ".plot-background { fill: #fbfcfe; stroke: #d8e0eb; }",
        ".figure-title { font-size: 25px; font-weight: 700; letter-spacing: -0.3px; }",
        ".figure-subtitle { font-size: 13px; fill: #64748b; }",
        ".panel-title { font-size: 18px; font-weight: 700; }",
        ".panel-subtitle { font-size: 12px; fill: #64748b; }",
        ".axis-label { font-size: 13px; font-weight: 600; fill: #475569; }",
        ".tick { font-size: 11px; fill: #64748b; }",
        ".legend { font-size: 11px; font-weight: 600; }",
        ".muted { fill: #64748b; }",
        ".grid { stroke: #dfe6ef; stroke-width: 1; }",
        ".grid.vertical { stroke: #edf1f6; }",
        ".series-line { stroke-width: 2.7; stroke-linecap: round; "
        "stroke-linejoin: round; }",
        ".reference { stroke: #94a3b8; stroke-width: 2; "
        "stroke-dasharray: 7 6; }",
        "</style>",
        "</defs>",
        f'<rect width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" class="canvas"/>',
        svg_text(
            CANVAS_WIDTH / 2,
            39,
            title,
            "figure-title",
        ),
        svg_text(
            CANVAS_WIDTH / 2,
            65,
            "Manufactured solution  ·  log–log scale  ·  static reference",
            "figure-subtitle",
        ),
    ]
    elements.extend(
        draw_panel(
            spatial,
            series,
            "Spatial refinement",
            "Grid spacing decreases while Δt is fixed",
            "h",
            "Grid spacing  h",
            "second-order  O(h²)",
            40,
            "spatial",
        )
    )
    elements.extend(
        draw_panel(
            temporal,
            series,
            "Temporal refinement",
            "Time step decreases while the grid is fixed",
            "dt",
            "Time step  Δt",
            "second-order  O(Δt²)",
            610,
            "temporal",
        )
    )
    elements.append("</svg>")
    output_path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rows = load_results(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    velocity_path = args.output_dir / "velocity.svg"
    pressure_path = args.output_dir / "pressure.svg"
    write_figure(
        rows,
        velocity_path,
        "Velocity convergence",
        [("L2_ux", "uₓ"), ("L2_uy", "uᵧ"), ("L2_uz", "u_z")],
    )
    write_figure(
        rows,
        pressure_path,
        "Pressure convergence",
        [("L2_p", "mean-aligned pressure")],
    )

    print(f"Velocity figure written to {velocity_path}")
    print(f"Pressure figure written to {pressure_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"could not generate convergence figures: {error}") from error
