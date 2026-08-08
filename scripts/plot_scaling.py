#!/usr/bin/env python3
"""Disegna i grafici dello scaling a partire dai CSV di run_scaling.sh.

Come plot_convergence.py, l'SVG viene scritto a mano: niente librerie da
installare, e il file resta leggibile.
"""

import argparse
import csv
from pathlib import Path

BLU = "#2563eb"
ROSSO = "#e11d48"
GRIGIO = "#94a3b8"
SCURO = "#1e293b"

LARGHEZZA = 1180
ALTEZZA = 400
PANNELLO = 360
MARGINE_X = 62
MARGINE_Y = 56
GRAFICO_W = 285
GRAFICO_H = 250


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", type=Path,
                        default=root / "build" / "scaling")
    parser.add_argument("output", nargs="?", type=Path,
                        default=root / "docs" / "scaling" / "scaling.svg")
    return parser.parse_args()


def load(path):
    if not path.exists():
        raise SystemExit(f"manca {path}\nLancia prima ./scripts/run_scaling.sh")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for row in rows:
        out.setdefault(row["study"], []).append(
            (int(row["procs"]), float(row["wall_ms"]), float(row["mpi_ms"])))
    for series in out.values():
        series.sort()
    return out


def asse_x(procs, x0):
    """I conteggi di processi raddoppiano: li spaziamo in modo uniforme."""
    passi = [1, 2, 4, 8, 16, 32]
    return x0 + GRAFICO_W * passi.index(procs) / (len(passi) - 3)


def pannello(parts, indice, titolo, sotto, y_max, y_label):
    x0 = MARGINE_X + indice * PANNELLO
    y0 = MARGINE_Y
    parts.append(f'<text x="{x0}" y="{y0 - 26}" class="titolo">{titolo}</text>')
    parts.append(f'<text x="{x0}" y="{y0 - 10}" class="sotto">{sotto}</text>')
    parts.append(f'<rect x="{x0}" y="{y0}" width="{GRAFICO_W}" '
                 f'height="{GRAFICO_H}" class="riquadro"/>')
    for frazione in (0, 0.25, 0.5, 0.75, 1.0):
        y = y0 + GRAFICO_H * (1 - frazione)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + GRAFICO_W}" '
                     f'y2="{y:.1f}" class="griglia"/>')
        parts.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="tacca-y">'
                     f'{frazione * y_max:.0f}</text>')
    parts.append(f'<text x="{x0 - 44}" y="{y0 + GRAFICO_H / 2}" '
                 f'class="asse" transform="rotate(-90 {x0 - 44} '
                 f'{y0 + GRAFICO_H / 2})">{y_label}</text>')
    parts.append(f'<text x="{x0 + GRAFICO_W / 2}" y="{y0 + GRAFICO_H + 40}" '
                 f'class="asse">processi</text>')
    for procs in (1, 2, 4, 8):
        x = asse_x(procs, x0)
        parts.append(f'<text x="{x:.1f}" y="{y0 + GRAFICO_H + 20}" '
                     f'class="tacca-x">{procs}</text>')
    return x0, y0


def curva(parts, x0, y0, punti, y_max, colore, tratteggio=""):
    coords = []
    for procs, valore in punti:
        x = asse_x(procs, x0)
        y = y0 + GRAFICO_H * (1 - min(valore, y_max) / y_max)
        coords.append((x, y))
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                    for i, (x, y) in enumerate(coords))
    parts.append(f'<path d="{path}" fill="none" stroke="{colore}" '
                 f'stroke-width="2.2" {tratteggio}/>')
    for x, y in coords:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" '
                     f'fill="{colore}"/>')


def legenda(parts, x0, y0, voci):
    for i, (colore, testo, tratteggio) in enumerate(voci):
        y = y0 + 14 + i * 17
        parts.append(f'<line x1="{x0 + 12}" y1="{y}" x2="{x0 + 38}" y2="{y}" '
                     f'stroke="{colore}" stroke-width="2.2" {tratteggio}/>')
        parts.append(f'<text x="{x0 + 44}" y="{y + 4}" class="legenda">'
                     f'{testo}</text>')


def barre(parts, x0, y0, serie, y_max):
    """Calcolo e comunicazione impilati, per vedere dove va il tempo."""
    larghezza = 34
    for procs, wall, mpi in serie:
        x = asse_x(procs, x0) - larghezza / 2
        h_tot = GRAFICO_H * min(wall, y_max) / y_max
        h_mpi = GRAFICO_H * min(mpi, y_max) / y_max
        parts.append(f'<rect x="{x:.1f}" y="{y0 + GRAFICO_H - h_tot:.1f}" '
                     f'width="{larghezza}" height="{h_tot - h_mpi:.1f}" '
                     f'fill="{BLU}" opacity="0.85"/>')
        parts.append(f'<rect x="{x:.1f}" y="{y0 + GRAFICO_H - h_mpi:.1f}" '
                     f'width="{larghezza}" height="{h_mpi:.1f}" '
                     f'fill="{ROSSO}" opacity="0.9"/>')
        parts.append(f'<text x="{x + larghezza / 2:.1f}" '
                     f'y="{y0 + GRAFICO_H - h_tot - 6:.1f}" class="valore">'
                     f'{100 * mpi / wall:.0f}%</text>')


def main():
    args = parse_args()
    simd = load(args.results_dir / "results.csv")
    scalare = load(args.results_dir / "results_scalar.csv")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{LARGHEZZA}" '
        f'height="{ALTEZZA}" viewBox="0 0 {LARGHEZZA} {ALTEZZA}">',
        '<style>',
        'text{font-family:"DejaVu Sans",sans-serif;fill:#1e293b}',
        '.titolo{font-size:15px;font-weight:600}',
        '.sotto{font-size:11px;fill:#64748b}',
        '.asse{font-size:12px;fill:#475569;text-anchor:middle}',
        '.tacca-x{font-size:11px;fill:#475569;text-anchor:middle}',
        '.tacca-y{font-size:11px;fill:#475569;text-anchor:end}',
        '.legenda{font-size:11px;fill:#334155}',
        '.valore{font-size:10px;fill:#475569;text-anchor:middle}',
        '.riquadro{fill:#f8fafc;stroke:#cbd5e1}',
        '.griglia{stroke:#e2e8f0;stroke-width:1}',
        '</style>',
        f'<rect width="{LARGHEZZA}" height="{ALTEZZA}" fill="white"/>',
    ]

    # 1. strong scaling: quanto si accorcia il tempo
    x0, y0 = pannello(parts, 0, "Strong scaling",
                      "problema fisso 128x128x128", 8, "speedup")
    curva(parts, x0, y0, [(p, p) for p, _, _ in scalare["strong"]], 8,
          GRIGIO, 'stroke-dasharray="5,4"')
    base = scalare["strong"][0][1]
    curva(parts, x0, y0, [(p, base / w) for p, w, _ in scalare["strong"]], 8,
          BLU)
    base = simd["strong"][0][1]
    curva(parts, x0, y0, [(p, base / w) for p, w, _ in simd["strong"]], 8,
          ROSSO)
    legenda(parts, x0, y0, [(GRIGIO, "ideale", 'stroke-dasharray="5,4"'),
                            (BLU, "scalare", ""), (ROSSO, "con SIMD", "")])

    # 2. weak scaling: il tempo dovrebbe restare costante
    x0, y0 = pannello(parts, 1, "Weak scaling",
                      "64x64x64 per processo", 100, "efficienza  %")
    curva(parts, x0, y0, [(p, 100) for p, _, _ in scalare["weak"]], 100,
          GRIGIO, 'stroke-dasharray="5,4"')
    base = scalare["weak"][0][1]
    curva(parts, x0, y0, [(p, 100 * base / w) for p, w, _ in scalare["weak"]],
          100, BLU)
    base = simd["weak"][0][1]
    curva(parts, x0, y0, [(p, 100 * base / w) for p, w, _ in simd["weak"]],
          100, ROSSO)
    legenda(parts, x0, y0, [(GRIGIO, "ideale", 'stroke-dasharray="5,4"'),
                            (BLU, "scalare", ""), (ROSSO, "con SIMD", "")])

    # 3. dove va il tempo
    y_max = max(w for _, w, _ in scalare["strong"]) * 1.15
    x0, y0 = pannello(parts, 2, "Dove va il tempo",
                      "strong scaling, scalare", y_max, "ms per passo")
    barre(parts, x0, y0, scalare["strong"], y_max)
    legenda(parts, x0, y0, [(BLU, "calcolo", ""), (ROSSO, "dentro MPI", "")])

    parts.append("</svg>")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts), encoding="utf-8")
    print(f"scritto {args.output}")


if __name__ == "__main__":
    main()
