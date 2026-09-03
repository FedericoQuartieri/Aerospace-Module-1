#!/usr/bin/env python3
"""I grafici dello studio di scaling, dal CSV unico di scripts/run_study.sh.

Come plot_scaling.py e plot_convergence.py: l'SVG viene scritto a mano, senza
librerie da installare. Sul cluster non c'e' matplotlib, e un grafico che si
puo' produrre solo altrove e' un grafico che non si guarda.

    ./scripts/run_study.sh merge          lo chiama da solo
    ./scripts/plot_study.py build/study/all.csv
"""

import argparse
import csv
import math
from pathlib import Path

BLU = "#2563eb"
ROSSO = "#e11d48"
VERDE = "#059669"
AMBRA = "#d97706"
VIOLA = "#7c3aed"
GRIGIO = "#94a3b8"
COLORI = [BLU, ROSSO, VERDE, AMBRA, VIOLA, "#0891b2", "#be185d"]

W, H = 300, 250          # area di disegno di un pannello
ML, MT = 74, 58          # margini attorno
MR, MB = 26, 62


def nice_step(raw):
    """Il passo `tondo` piu' vicino: 1, 2, 2.5 o 5 per la potenza di dieci."""
    if raw <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        if raw <= factor * magnitude:
            return factor * magnitude
    return 10 * magnitude


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path,
                        default=root / "build" / "study" / "all.csv")
    parser.add_argument("-o", "--outdir", type=Path,
                        default=root / "docs" / "scaling" / "study")
    return parser.parse_args()


def load(path):
    if not path.exists():
        raise SystemExit(f"manca {path}\nLancia prima ./scripts/run_study.sh merge")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("status") == "ok"]
    for row in rows:
        for key in ("batch", "simd", "omp", "mpi", "ranks", "threads",
                    "nx", "ny", "nz", "steps", "px", "py", "pz"):
            row[key] = int(row[key] or 0)
        for key in ("wall_ms", "mpi_ms", "eta_ms", "zeta_ms", "u_ms",
                    "untimed_ms", "rss_mb"):
            row[key] = float(row[key] or 0)
    return rows


def load_ceilings(csv_path):
    path = csv_path.parent / "01_ceiling" / "ceilings.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(r, threads=int(r["threads"]), rate=float(r["rate"]))
                for r in csv.DictReader(handle)]


# --------------------------------------------------------------- il pannello


class Panel:
    """Un riquadro con assi. Le x sono sempre logaritmiche in base 2 quando
    contano unita' di calcolo: raddoppiano, e su scala lineare le prime cinque
    finirebbero tutte addosso all'origine."""

    def __init__(self, parts, col, row, title, subtitle, xlabel, ylabel,
                 xs, ys, xlog=True, ylog=False, ymin=0.0):
        self.parts = parts
        self.x0 = ML + col * (W + ML + MR)
        self.y0 = MT + row * (H + MT + MB)
        self.xlog, self.ylog = xlog, ylog
        xs = [x for x in xs if x > 0] or [1]
        ys = [y for y in ys if y > 0 or not ylog] or [1]
        self.xmin, self.xmax = min(xs), max(xs)
        if self.xmax == self.xmin:
            self.xmax = self.xmin * 2
        self.ymax = max(ys) * 1.08
        self.ymin = min(ys) / 1.3 if ylog else ymin
        if self.ymax <= self.ymin:
            self.ymax = self.ymin + 1

        parts.append(f'<text x="{self.x0}" y="{self.y0 - 30}" class="titolo">'
                     f'{title}</text>')
        parts.append(f'<text x="{self.x0}" y="{self.y0 - 14}" class="sotto">'
                     f'{subtitle}</text>')
        parts.append(f'<rect x="{self.x0}" y="{self.y0}" width="{W}" '
                     f'height="{H}" class="riquadro"/>')
        parts.append(f'<text x="{self.x0 - 52}" y="{self.y0 + H / 2}" '
                     f'class="asse" transform="rotate(-90 {self.x0 - 52} '
                     f'{self.y0 + H / 2})">{ylabel}</text>')
        parts.append(f'<text x="{self.x0 + W / 2}" y="{self.y0 + H + 42}" '
                     f'class="asse">{xlabel}</text>')
        self._grid_y()

    def _grid_y(self):
        if self.ylog:
            lo = math.floor(math.log10(self.ymin))
            hi = math.ceil(math.log10(self.ymax))
            values = [10 ** e for e in range(int(lo), int(hi) + 1)]
        else:
            # Tacche su numeri tondi: una scala che dice 2.16 e 1.62 si legge
            # peggio di una che dice 2 e 1.5, e il grafico non guadagna niente
            # dalla precisione dell'estremo.
            step = nice_step((self.ymax - self.ymin) / 4)
            self.ymax = math.ceil(self.ymax / step) * step
            values = [self.ymin + step * i
                      for i in range(int((self.ymax - self.ymin) / step) + 1)]
        for value in values:
            if not (self.ymin <= value <= self.ymax):
                continue
            y = self.py(value)
            self.parts.append(f'<line x1="{self.x0}" y1="{y:.1f}" '
                              f'x2="{self.x0 + W}" y2="{y:.1f}" class="griglia"/>')
            label = f"{value:g}"
            self.parts.append(f'<text x="{self.x0 - 8}" y="{y + 4:.1f}" '
                              f'class="tacca-y">{label}</text>')

    def xticks(self, values, labels=None):
        labels = labels or [str(v) for v in values]
        for value, label in zip(values, labels):
            x = self.px(value)
            self.parts.append(f'<text x="{x:.1f}" y="{self.y0 + H + 20}" '
                              f'class="tacca-x">{label}</text>')
            self.parts.append(f'<line x1="{x:.1f}" y1="{self.y0 + H}" '
                              f'x2="{x:.1f}" y2="{self.y0 + H + 5}" '
                              f'class="griglia"/>')

    def px(self, x):
        if self.xlog:
            span = math.log2(self.xmax) - math.log2(self.xmin) or 1
            f = (math.log2(max(x, 1e-9)) - math.log2(self.xmin)) / span
        else:
            span = self.xmax - self.xmin or 1
            f = (x - self.xmin) / span
        return self.x0 + W * min(max(f, 0), 1)

    def py(self, y):
        if self.ylog:
            span = math.log10(self.ymax) - math.log10(self.ymin) or 1
            f = (math.log10(max(y, 1e-12)) - math.log10(self.ymin)) / span
        else:
            span = self.ymax - self.ymin or 1
            f = (y - self.ymin) / span
        return self.y0 + H * (1 - min(max(f, 0), 1))

    def line(self, points, colore, dash="", marker=True):
        points = [p for p in points if p[1] is not None]
        if not points:
            return
        coords = [(self.px(x), self.py(y)) for x, y in points]
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                        for i, (x, y) in enumerate(coords))
        dash = f'stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path d="{path}" fill="none" stroke="{colore}" '
                          f'stroke-width="2.2" {dash}/>')
        if marker:
            for x, y in coords:
                self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" '
                                  f'fill="{colore}"/>')

    def bars(self, labels, values, colori):
        step = W / max(len(values), 1)
        width = step * 0.62
        for i, (label, value) in enumerate(zip(labels, values)):
            x = self.x0 + step * (i + 0.5) - width / 2
            top = self.py(value)
            self.parts.append(f'<rect x="{x:.1f}" y="{top:.1f}" '
                              f'width="{width:.1f}" '
                              f'height="{self.y0 + H - top:.1f}" '
                              f'fill="{colori[i % len(colori)]}" opacity="0.85"/>')
            self.parts.append(f'<text x="{x + width / 2:.1f}" '
                              f'y="{top - 5:.1f}" class="valore">'
                              f'{value:.0f}</text>')
            self.parts.append(
                f'<text x="{x + width / 2:.1f}" y="{self.y0 + H + 16}" '
                f'class="tacca-x" transform="rotate(-35 {x + width / 2:.1f} '
                f'{self.y0 + H + 16})">{label}</text>')

    def legend(self, voci, dx=14, dy=10):
        for i, (colore, testo, dash) in enumerate(voci):
            y = self.y0 + dy + 14 + i * 16
            dash = f'stroke-dasharray="{dash}"' if dash else ""
            self.parts.append(f'<line x1="{self.x0 + dx}" y1="{y}" '
                              f'x2="{self.x0 + dx + 24}" y2="{y}" '
                              f'stroke="{colore}" stroke-width="2.2" {dash}/>')
            self.parts.append(f'<text x="{self.x0 + dx + 30}" y="{y + 4}" '
                              f'class="legenda">{testo}</text>')


def svg(cols, rows, parts):
    width = ML + cols * (W + ML + MR)
    height = MT + rows * (H + MT + MB)
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text{font-family:"DejaVu Sans",sans-serif;fill:#1e293b}',
        '.titolo{font-size:15px;font-weight:600}',
        '.sotto{font-size:11px;fill:#64748b}',
        '.asse{font-size:12px;fill:#475569;text-anchor:middle}',
        '.tacca-x{font-size:10px;fill:#475569;text-anchor:middle}',
        '.tacca-y{font-size:10px;fill:#475569;text-anchor:end}',
        '.legenda{font-size:11px;fill:#334155}',
        '.valore{font-size:9px;fill:#475569;text-anchor:middle}',
        '.riquadro{fill:#f8fafc;stroke:#cbd5e1}',
        '.griglia{stroke:#e2e8f0;stroke-width:1}',
        '</style>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]
    return "\n".join(head + parts + ["</svg>"])


def write(outdir, name, cols, rows, parts):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    path.write_text(svg(cols, rows, parts), encoding="utf-8")
    print(f"  {path}")


def pick(rows, **filters):
    out = []
    for row in rows:
        if all(row.get(k) == v for k, v in filters.items()):
            out.append(row)
    return out


def series(rows, xkey, ykey="wall_ms"):
    points = sorted((row[xkey], row[ykey]) for row in rows)
    return points


# ------------------------------------------------------------------ i grafici


def fig_ceiling(ceilings, outdir):
    if not ceilings:
        return
    parts = []
    for col, kind in enumerate(("trig", "triad")):
        data = sorted((c["threads"], c["rate"]) for c in ceilings
                      if c["kind"] == kind)
        if not data:
            continue
        base = data[0][1]
        speedup = [(t, r / base) for t, r in data]
        ideal = [(t, t) for t, _ in data]
        panel = Panel(parts, col, 0,
                      "Tetto di calcolo" if kind == "trig" else "Tetto di banda",
                      "trigonometria pura, nessuna memoria" if kind == "trig"
                      else "triad: due letture e una scrittura per elemento",
                      "thread", "speedup",
                      [t for t, _ in data], [s for _, s in speedup] + [1],
                      xlog=True)
        panel.xticks([t for t, _ in data])
        panel.line(ideal, GRIGIO, dash="5,4", marker=False)
        panel.line(speedup, BLU if kind == "trig" else ROSSO)
        panel.legend([(GRIGIO, "ideale", "5,4"),
                      (BLU if kind == "trig" else ROSSO, "misurato", "")])
    write(outdir, "01-tetti.svg", 2, 1, parts)


def fig_threads(rows, outdir):
    data = pick(rows, phase="02_threads")
    data = [r for r in data if "socket" not in r["label"]]
    grids = sorted({r["nx"] for r in data})
    if not grids:
        return
    parts = []
    for col, grid in enumerate(grids):
        here = [r for r in data if r["nx"] == grid]
        ys = [r["wall_ms"] for r in here]
        xs = [r["threads"] for r in here]
        panel = Panel(parts, col, 0, f"Thread, {grid}^3",
                      "un processo, nessun asse diviso",
                      "thread", "ms per passo", xs, ys, xlog=True, ylog=True)
        panel.xticks(sorted(set(xs)))
        voci = []
        for i, backend in enumerate(("schur", "pipeline")):
            points = series([r for r in here if r["backend"] == backend],
                            "threads")
            if not points:
                continue
            panel.line(points, COLORI[i])
            voci.append((COLORI[i], backend, ""))
            base = points[0][1]
            panel.line([(x, base / x) for x, _ in points], GRIGIO,
                       dash="5,4", marker=False)
        voci.append((GRIGIO, "ideale", "5,4"))
        panel.legend(voci)
    write(outdir, "02-thread.svg", len(grids), 1, parts)


def fig_mpi(rows, outdir):
    data = [r for r in pick(rows, phase="03_mpi") if "ponte" not in r["label"]]
    combos = sorted({(r["nx"], r["simd"]) for r in data})
    if not combos:
        return
    parts = []
    for col, (grid, simd) in enumerate(combos):
        here = [r for r in data if r["nx"] == grid and r["simd"] == simd]
        xs = [r["ranks"] for r in here]
        panel = Panel(parts, col, 0, f"Processi, {grid}^3",
                      f"un thread per rank, simd={simd}",
                      "processi", "speedup", xs, xs, xlog=True)
        panel.xticks(sorted(set(xs)))
        voci = [(GRIGIO, "ideale", "5,4")]
        panel.line([(x, x) for x in sorted(set(xs))], GRIGIO, dash="5,4",
                   marker=False)
        for i, backend in enumerate(("schur", "pipeline")):
            points = series([r for r in here if r["backend"] == backend],
                            "ranks")
            if not points:
                continue
            base = points[0][1]
            panel.line([(x, base / y) for x, y in points], COLORI[i])
            voci.append((COLORI[i], backend, ""))
        panel.legend(voci)
    write(outdir, "03-mpi.svg", len(combos), 1, parts)


def fig_shape(rows, outdir):
    data = pick(rows, phase="04_shape")
    groups = sorted({(r["ranks"], r["simd"]) for r in data})
    if not groups:
        return
    parts = []
    for col, (ranks, simd) in enumerate(groups):
        here = [r for r in data if r["ranks"] == ranks and r["simd"] == simd]
        forms = sorted({(r["px"], r["py"], r["pz"]) for r in here})
        labels, values, colori = [], [], []
        for form in forms:
            for i, backend in enumerate(("schur", "pipeline")):
                match = [r for r in here
                         if (r["px"], r["py"], r["pz"]) == form
                         and r["backend"] == backend]
                if not match:
                    continue
                labels.append(f"{'x'.join(str(v) for v in form)} {backend[:4]}")
                values.append(match[0]["wall_ms"])
                colori.append(COLORI[i])
        if not values:
            continue
        panel = Panel(parts, col, 0, f"Forma, {ranks} processi",
                      f"stesso blocco per processo, simd={simd}",
                      "forma della griglia di processi", "ms per passo",
                      [1], values)
        panel.bars(labels, values, colori)
        panel.legend([(COLORI[0], "schur", ""), (COLORI[1], "pipeline", "")])
    write(outdir, "04-forma.svg", len(groups), 1, parts)


def fig_hybrid(rows, outdir):
    data = pick(rows, phase="05_hybrid")
    groups = sorted({(r["nx"], r["ranks"] * r["threads"]) for r in data})
    if not groups:
        return
    cols = min(len(groups), 3)
    parts = []
    for index, (grid, product) in enumerate(groups):
        here = [r for r in data
                if r["nx"] == grid and r["ranks"] * r["threads"] == product]
        xs = [r["ranks"] for r in here]
        ys = [r["wall_ms"] for r in here]
        panel = Panel(parts, index % cols, index // cols,
                      f"{grid}^3, rank x thread = {product}",
                      "stessi core, spesi diversamente",
                      "processi (thread = prodotto / processi)",
                      "ms per passo", xs, ys, xlog=True, ylog=True)
        panel.xticks(sorted(set(xs)))
        voci = []
        for i, backend in enumerate(("schur", "pipeline")):
            points = series([r for r in here if r["backend"] == backend],
                            "ranks")
            if not points:
                continue
            panel.line(points, COLORI[i])
            voci.append((COLORI[i], backend, ""))
        panel.legend(voci)
    write(outdir, "05-ibrido.svg", cols, (len(groups) + cols - 1) // cols, parts)


def fig_batch(rows, outdir):
    data = [r for r in pick(rows, phase="06_batch")
            if r["backend"] == "pipeline"]
    if not data:
        return
    parts = []
    shapes = sorted({(r["ranks"], r["px"], r["py"], r["pz"]) for r in data})
    xs = [r["batch"] for r in data]
    ys = [r["wall_ms"] for r in data]
    panel = Panel(parts, 0, 0, "Il batch della pipeline",
                  "PIPELINE_BATCH_LINES, l'unica manopola",
                  "linee per batch", "ms per passo", xs, ys,
                  xlog=True, ylog=True)
    panel.xticks(sorted(set(xs)))
    voci = []
    for i, (ranks, px, py, pz) in enumerate(shapes):
        points = series([r for r in data
                         if (r["ranks"], r["px"], r["py"], r["pz"])
                         == (ranks, px, py, pz)], "batch")
        if len(points) < 2:
            continue
        panel.line(points, COLORI[i % len(COLORI)])
        voci.append((COLORI[i % len(COLORI)], f"{ranks}p {px}x{py}x{pz}", ""))
    panel.legend(voci)

    rss = [r["rss_mb"] for r in data if r["rss_mb"] > 0]
    if rss:
        panel2 = Panel(parts, 1, 0, "Il prezzo in memoria",
                       "c' e d' di tutto il blocco locale, arrotondati a batch interi",
                       "linee per batch", "MB per processo",
                       xs, rss, xlog=True)
        panel2.xticks(sorted(set(xs)))
        voci = []
        for i, (ranks, px, py, pz) in enumerate(shapes):
            points = series([r for r in data
                             if (r["ranks"], r["px"], r["py"], r["pz"])
                             == (ranks, px, py, pz)], "batch", "rss_mb")
            if len(points) < 2:
                continue
            panel2.line(points, COLORI[i % len(COLORI)])
            voci.append((COLORI[i % len(COLORI)], f"{ranks}p {px}x{py}x{pz}", ""))
        panel2.legend(voci)
    write(outdir, "06-batch.svg", 2, 1, parts)


def fig_weak(rows, outdir):
    data = pick(rows, phase="07_weak")
    if not data:
        return
    parts = []
    for col, kind in enumerate(("proc", "thread")):
        here = [r for r in data
                if (r["ranks"] > 1 if kind == "proc" else r["threads"] > 1)
                or (r["ranks"] == 1 and r["threads"] == 1)]
        if not here:
            continue
        key = "ranks" if kind == "proc" else "threads"
        xs = [max(r["ranks"], r["threads"]) for r in here]
        panel = Panel(parts, col, 0,
                      "Weak scaling, processi" if kind == "proc"
                      else "Weak scaling, thread",
                      "stesso blocco per unita', problema che cresce",
                      "unita' di calcolo", "efficienza  %", xs, [0, 110],
                      xlog=True, ymin=0)
        panel.xticks(sorted(set(xs)))
        panel.line([(x, 100) for x in sorted(set(xs))], GRIGIO, dash="5,4",
                   marker=False)
        voci = [(GRIGIO, "ideale", "5,4")]
        for i, backend in enumerate(("schur", "pipeline")):
            points = sorted((max(r["ranks"], r["threads"]), r["wall_ms"])
                            for r in here if r["backend"] == backend)
            if not points:
                continue
            base = points[0][1]
            panel.line([(x, 100 * base / y) for x, y in points], COLORI[i])
            voci.append((COLORI[i], backend, ""))
        panel.legend(voci)
    write(outdir, "07-weak.svg", 2, 1, parts)


def fig_size(rows, outdir):
    data = pick(rows, phase="08_size")
    if not data:
        return

    def config(row):
        if row["mpi"] == 0:
            return "seriale"
        return f"1 x {row['threads']}" if row["ranks"] == 1 else f"{row['ranks']} x 1"

    parts = []
    xs = [r["nx"] for r in data]
    ys = [r["wall_ms"] * 1e6 / (r["nx"] * r["ny"] * r["nz"]) for r in data]
    panel = Panel(parts, 0, 0, "Costo per cella", "a risorse fisse",
                  "lato della griglia", "ns per cella-passo", xs, ys,
                  xlog=True, ylog=True)
    panel.xticks(sorted(set(xs)))
    voci = []
    configs = sorted({config(r) for r in data})
    for i, name in enumerate(configs):
        for j, backend in enumerate(("schur", "pipeline")):
            points = sorted(
                (r["nx"], r["wall_ms"] * 1e6 / (r["nx"] * r["ny"] * r["nz"]))
                for r in data if config(r) == name and r["backend"] == backend)
            if not points:
                continue
            colore = COLORI[(i * 2 + j) % len(COLORI)]
            panel.line(points, colore, dash="" if backend == "schur" else "4,3")
            voci.append((colore, f"{backend} {name}", ""))
    panel.legend(voci)
    write(outdir, "08-taglia.svg", 1, 1, parts)


def main():
    args = parse_args()
    rows = load(args.csv)
    ceilings = load_ceilings(args.csv)
    print(f"{len(rows)} misure valide da {args.csv}")
    fig_ceiling(ceilings, args.outdir)
    fig_threads(rows, args.outdir)
    fig_mpi(rows, args.outdir)
    fig_shape(rows, args.outdir)
    fig_hybrid(rows, args.outdir)
    fig_batch(rows, args.outdir)
    fig_weak(rows, args.outdir)
    fig_size(rows, args.outdir)


if __name__ == "__main__":
    main()
