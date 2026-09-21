#!/usr/bin/env python3
"""The plots of the exhaustive campaign (phases 10-15), from the merged CSV.

    ./scripts/run_study.sh merge          produces build/study/all.csv
    ./scripts/plot_matrix.py              draws into docs/scaling/matrix/
    ./scripts/plot_matrix.py build/study/all.csv -o /tmp/figure

Like plot_scaling.py and plot_convergence.py: the SVG is written by hand, with
no libraries to install. There is no matplotlib on the cluster, and a plot that
can only be produced somewhere else is a plot nobody looks at.

The colours instead are new, and not chosen by taste: the categorical scale is
checked for colour blindness (OKLab separation >= 8 between two neighbouring
colours, >= 15 to normal vision) and every series also carries its label next
to it, because two of the hues fall below the 3:1 contrast ratio on the light
background. The figures adapt to the viewer's dark theme.

One rule, and it holds for all of them: never two y axes in the same panel.
Aligning two different scales is arbitrary and invents a correlation the data
does not have. Two quantities, two panels.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

# ----------------------------------------------------------------- panels
#
# Geometry of the panels: margins, ticks, logarithmic scales, bars. It used to
# be in plot_study.py, the script of the ten-phase study; that one has been
# removed together with the phases it drew, and the class has moved here.

W, H = 300, 250          # drawing area of a panel
ML, MT = 74, 58          # margins around it
MR, MB = 26, 62


def nice_step(raw):
    """The nearest `round` step: 1, 2, 2.5 or 5 times the power of ten."""
    if raw <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        if raw <= factor * magnitude:
            return factor * magnitude
    return 10 * magnitude


class Panel:
    """A panel with axes. The x axes are always base-2 logarithmic when they
    count compute units: they double, and on a linear scale the first five
    would all pile up on the origin."""

    def __init__(self, parts, col, row, title, subtitle, xlabel, ylabel,
                 xs, ys, xlog=True, ylog=False, ymin=0.0):
        self.parts = parts
        self.x0 = ML + col * (W + ML + MR)
        self.y0 = MT + row * (H + MT + MB)
        self.xlog, self.ylog = xlog, ylog
        # Where the panel is already occupied, in pixels: lines, markers,
        # labels. The legend and the series labels look at it to choose a place
        # that does not cover the data.
        self.occupati = []
        xs = [x for x in xs if x > 0] or [1]
        ys = [y for y in ys if y > 0 or not ylog] or [1]
        self.xmin, self.xmax = min(xs), max(xs)
        if self.xmax == self.xmin:
            self.xmax = self.xmin * 2
        self.ymax = max(ys) * 1.08
        self.ymin = min(ys) / 1.3 if ylog else ymin
        if self.ymax <= self.ymin:
            self.ymax = self.ymin + 1

        parts.append(f'<text x="{self.x0}" y="{self.y0 - 30}" class="title">'
                     f'{title}</text>')
        parts.append(f'<text x="{self.x0}" y="{self.y0 - 14}" class="subtitle">'
                     f'{subtitle}</text>')
        parts.append(f'<rect x="{self.x0}" y="{self.y0}" width="{W}" '
                     f'height="{H}" class="panel"/>')
        parts.append(f'<text x="{self.x0 - 52}" y="{self.y0 + H / 2}" '
                     f'class="axis" transform="rotate(-90 {self.x0 - 52} '
                     f'{self.y0 + H / 2})">{ylabel}</text>')
        parts.append(f'<text x="{self.x0 + W / 2}" y="{self.y0 + H + 42}" '
                     f'class="axis">{xlabel}</text>')
        self._grid_y()

    def _grid_y(self):
        if self.ylog:
            lo = int(math.floor(math.log10(self.ymin)))
            hi = int(math.ceil(math.log10(self.ymax)))
            # Powers of ten alone leave an axis with one number, or none, when
            # the data lie within a decade: then it falls back to 1-2-5, and if
            # that is not enough to all the multiples.
            values = []
            for multipli in ((1,), (1, 2, 5), tuple(range(1, 10))):
                values = [m * 10 ** e for e in range(lo, hi + 1)
                          for m in multipli]
                if sum(self.ymin <= v <= self.ymax for v in values) >= 3:
                    break
        else:
            # Ticks on round numbers: a scale that says 2.16 and 1.62 reads
            # worse than one that says 2 and 1.5, and the plot gains nothing
            # from the precision of the endpoint.
            step = nice_step((self.ymax - self.ymin) / 4)
            self.ymax = math.ceil(self.ymax / step) * step
            values = [self.ymin + step * i
                      for i in range(int((self.ymax - self.ymin) / step) + 1)]
        ultima = None
        for value in values:
            if not (self.ymin <= value <= self.ymax):
                continue
            y = self.py(value)
            # Two labels less than 13 pixels apart overlap: the second is
            # skipped, together with its grid line.
            if ultima is not None and abs(y - ultima) < 13:
                continue
            ultima = y
            self.parts.append(f'<line x1="{self.x0}" y1="{y:.1f}" '
                              f'x2="{self.x0 + W}" y2="{y:.1f}" class="grid"/>')
            label = f"{value:g}"
            self.parts.append(f'<text x="{self.x0 - 8}" y="{y + 4:.1f}" '
                              f'class="tick-y">{label}</text>')

    def xticks(self, values, labels=None):
        labels = labels or [str(v) for v in values]
        for value, label in zip(values, labels):
            x = self.px(value)
            self.parts.append(f'<text x="{x:.1f}" y="{self.y0 + H + 20}" '
                              f'class="tick-x">{label}</text>')
            self.parts.append(f'<line x1="{x:.1f}" y1="{self.y0 + H}" '
                              f'x2="{x:.1f}" y2="{self.y0 + H + 5}" '
                              f'class="grid"/>')

    def px_raw(self, x):
        """The x in pixels without clamping it at the edge: it is needed to
        clip the lines."""
        if self.xlog:
            span = math.log2(self.xmax) - math.log2(self.xmin) or 1
            f = (math.log2(max(x, 1e-9)) - math.log2(self.xmin)) / span
        else:
            span = self.xmax - self.xmin or 1
            f = (x - self.xmin) / span
        return self.x0 + W * f

    def py_raw(self, y):
        if self.ylog:
            span = math.log10(self.ymax) - math.log10(self.ymin) or 1
            f = (math.log10(max(y, 1e-12)) - math.log10(self.ymin)) / span
        else:
            span = self.ymax - self.ymin or 1
            f = (y - self.ymin) / span
        return self.y0 + H * (1 - f)

    def px(self, x):
        return min(max(self.px_raw(x), self.x0), self.x0 + W)

    def py(self, y):
        return min(max(self.py_raw(y), self.y0), self.y0 + H)

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
                              f'y="{top - 5:.1f}" class="value">'
                              f'{value:.0f}</text>')
            # Right-aligned on the tick and rotated: the text descends to the
            # left of its bar instead of spreading on both sides, and does not
            # reach the axis title.
            self.parts.append(
                f'<text x="{x + width / 2:.1f}" y="{self.y0 + H + 14}" '
                f'class="tick-x" style="text-anchor:end" '
                f'transform="rotate(-35 {x + width / 2:.1f} '
                f'{self.y0 + H + 14})">{label}</text>')

    def legend(self, voci, dx=14, dy=10):
        for i, (colore, testo, dash) in enumerate(voci):
            y = self.y0 + dy + 14 + i * 16
            dash = f'stroke-dasharray="{dash}"' if dash else ""
            self.parts.append(f'<line x1="{self.x0 + dx}" y1="{y}" '
                              f'x2="{self.x0 + dx + 24}" y2="{y}" '
                              f'stroke="{colore}" stroke-width="2.2" {dash}/>')
            self.parts.append(f'<text x="{self.x0 + dx + 30}" y="{y + 4}" '
                              f'class="legend">{testo}</text>')

# -------------------------------------------------------------------- colours
#
# Categorical scale, in order: it is the order that guarantees the separation
# between close hues, so the series are assigned from the first onwards and are
# not reshuffled when a filter removes one. A ninth series does not exist: the
# figure is merged or split.

SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SERIE_SCURO = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
NEUTRO = "#8a8a83"

# Sequential scale for the heat maps: a single hue, from light to dark. Never a
# rainbow: hues have no natural order, values do.
RAMPA = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
         "#0d366b"]

STILE = """
svg{--background:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;
    --ink-3:#6f6e69;--panel:#f6f6f4;--line:#d9d8d2;--grid:#e8e7e2}
@media (prefers-color-scheme:dark){
svg{--background:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;
    --ink-3:#a3a299;--panel:#232322;--line:#3a3a37;--grid:#2e2e2c}}
text{font-family:"DejaVu Sans",sans-serif;fill:var(--ink)}
.title{font-size:15px;font-weight:600}
.subtitle{font-size:11px;fill:var(--ink-2)}
.axis{font-size:12px;fill:var(--ink-2);text-anchor:middle}
.tick-x{font-size:10px;fill:var(--ink-3);text-anchor:middle}
.tick-y{font-size:10px;fill:var(--ink-3);text-anchor:end}
.legend{font-size:11px;fill:var(--ink-2)}
.value{font-size:9px;fill:var(--ink-2);text-anchor:middle}
.cell{font-size:9px;text-anchor:middle}
.note{font-size:11px;fill:var(--ink-2)}
.huge{font-size:46px;font-weight:600}
.panel{fill:var(--panel);stroke:var(--line)}
.grid{stroke:var(--grid);stroke-width:1}
"""

# The dark hues are applied by rewriting the variable: one block for each
# series used, generated when needed.


def stile_serie(n):
    righe = []
    for i in range(n):
        righe.append(f".s{i}{{stroke:{SERIE[i]};fill:{SERIE[i]}}}")
    righe.append("@media (prefers-color-scheme:dark){")
    for i in range(n):
        righe.append(f".s{i}{{stroke:{SERIE_SCURO[i]};fill:{SERIE_SCURO[i]}}}")
    righe.append("}")
    return "\n".join(righe)


def svg(cols, rows, parts, serie=4, altezza_extra=0):
    width = ML + cols * (W + ML + MR)
    height = MT + rows * (H + MT + MB) + altezza_extra
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        "<style>", STILE, stile_serie(serie), "</style>",
        f'<rect width="{width}" height="{height}" fill="var(--background)"/>',
    ]
    return "\n".join(head + parts + ["</svg>"])


def write(outdir, name, cols, rows, parts, serie=4, altezza_extra=0):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    path.write_text(svg(cols, rows, parts, serie, altezza_extra),
                    encoding="utf-8")
    print(f"  {path}")


# --------------------------------------------------------------- the data


NUMERICHE = ("batch", "simd", "omp", "mpi", "ranks", "threads", "nx", "ny",
             "nz", "steps", "px", "py", "pz", "wall_ms", "mpi_ms", "eta_ms",
             "zeta_ms", "u_ms", "psi_ms", "philow_ms", "phihigh_ms",
             "pressure_ms", "porosity_ms", "untimed_ms", "cellstep_1e8s",
             "rss_mb", "l2_ux", "l2_p", "g_ms")


def load(path):
    """The successful rows, with the numbers already converted.

    A failed or timed-out case carries empty columns: keeping it would mean
    drawing a zero where there is no measurement, and it is the quickest way of
    reading a hole as a result."""
    if not path.exists():
        sys.exit(f"missing {path}\n  ./scripts/run_study.sh merge")

    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("status") != "ok":
                continue
            row = dict(raw)
            for key in NUMERICHE:
                value = raw.get(key, "")
                try:
                    row[key] = float(value) if value not in (None, "") else None
                except ValueError:
                    row[key] = None
            if row["wall_ms"] in (None, 0):
                continue
            # The two reference rows of each configuration are recognised by
            # the suffix that study_baseline appends to the label. Marking them
            # here costs once and allows pick() to keep them out of the curves.
            label = row.get("label") or ""
            if label.endswith(" serial"):
                row["baseline"] = "serial"
            elif label.endswith(" T(1)"):
                row["baseline"] = "T(1)"
            else:
                row["baseline"] = None
            rows.append(row)
    print(f"{len(rows)} valid measurements from {path}")
    return rows


def pick(rows, **filtri):
    """The rows that satisfy the filters, references excluded.

    Serial and T(1) are not points of a curve, they are the denominators: they
    have one rank and one thread, so without this they would end up inside
    every plot as if they were the single-process case -- which, however,
    already exists and is another one. To get them, pass baseline="serial" or
    baseline="T(1)"."""
    filtri.setdefault("baseline", None)
    out = []
    for row in rows:
        ok = True
        for key, value in filtri.items():
            if isinstance(value, (list, tuple, set)):
                ok = ok and row.get(key) in value
            else:
                ok = ok and row.get(key) == value
            if not ok:
                break
        if ok:
            out.append(row)
    return out


def valori(rows, key):
    return sorted({r[key] for r in rows if r[key] is not None})


def _conta(panel, riquadro, margine=2.0):
    """How many already-drawn points fall inside a panel (x, y, w, h)."""
    x, y, w, h = riquadro
    return sum(1 for a, b in panel.occupati
               if x - margine <= a <= x + w + margine
               and y - margine <= b <= y + h + margine)


def _occupa(panel, riquadro, passo=6.0):
    x, y, w, h = riquadro
    nx, ny = max(1, int(w / passo)), max(1, int(h / passo))
    for i in range(nx + 1):
        for j in range(ny + 1):
            panel.occupati.append((x + w * i / nx, y + h * j / ny))


def _campiona(panel, a, b, passo=5.0):
    (x1, y1), (x2, y2) = a, b
    n = max(1, int(math.hypot(x2 - x1, y2 - y1) / passo))
    for k in range(n + 1):
        t = k / n
        panel.occupati.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))


def _taglia(panel, a, b):
    """The segment a-b clipped to the panel (Liang-Barsky), or None."""
    (x1, y1), (x2, y2) = a, b
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x1 - panel.x0), (dx, panel.x0 + W - x1),
                 (-dy, y1 - panel.y0), (dy, panel.y0 + H - y1)):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    return (x1 + t0 * dx, y1 + t0 * dy), (x1 + t1 * dx, y1 + t1 * dy)


def _dentro(panel, x, y):
    return (panel.x0 - 0.5 <= x <= panel.x0 + W + 0.5
            and panel.y0 - 0.5 <= y <= panel.y0 + H + 0.5)


def etichetta_fine(parts, panel, points, colore, testo):
    """The label next to the last point of the line.

    Two of the four hues do not reach 3:1 contrast on the light background: the
    legend alone is not enough to tell them apart, and the label attached to
    the series is what makes the figure readable even when printed in black and
    white."""
    if not points:
        return
    x, y = points[-1]
    X, Y = panel.px(x), panel.py(y)
    larghezza = len(testo) * 6.2 + 4
    # Inside the panel, right-aligned on the last point: outside it would land
    # on the axis of the neighbouring panel, which is 26 pixels away. Above or
    # below the point, where it does not leave the panel and covers less data.
    def punteggio(base):
        fuori = base - 11 < panel.y0 + 2 or base + 3 > panel.y0 + H - 2
        return (fuori, _conta(panel, (X - 6 - larghezza, base - 11,
                                      larghezza, 14)))
    base = min((Y - 10, Y + 20), key=punteggio)
    parts.append(f'<text x="{X - 6:.1f}" y="{base:.1f}" '
                 f'class="legend" text-anchor="end" '
                 f'style="fill:{colore}">{testo}</text>')
    _occupa(panel, (X - 6 - larghezza, base - 11, larghezza, 14))


def piazzamento(r, t):
    """Singular/plural agreement: `1 process x 1 thread', not `1 processes'."""
    return (f"{r:.0f} process{'' if r == 1 else 'es'} x "
            f"{t:.0f} thread{'' if t == 1 else 's'}")


def linea(parts, panel, punti, colore, tratteggio="", marcatori=True):
    """A series. Thin stroke, 9px markers: below that size a point gets lost,
    above it the markers become the plot."""
    punti = [p for p in punti if p[1] is not None]
    if not punti:
        return
    grezze = [(panel.px_raw(x), panel.py_raw(y)) for x, y in punti]
    dash = f' stroke-dasharray="{tratteggio}"' if tratteggio else ""
    # A line that leaves the scale is clipped at the edge of the panel. Before,
    # it was squashed onto the edge, and a flat stretch along the axis looked
    # like a measurement: the ideal thread line seemed to stop at 14.
    tratti, corrente = [], []
    for a, b in zip(grezze, grezze[1:]):
        pezzo = _taglia(panel, a, b)
        if pezzo is None:
            if corrente:
                tratti.append(corrente)
            corrente = []
            continue
        p, q = pezzo
        if corrente and math.hypot(corrente[-1][0] - p[0],
                                   corrente[-1][1] - p[1]) > 0.01:
            tratti.append(corrente)
            corrente = []
        if not corrente:
            corrente = [p]
        corrente.append(q)
    if corrente:
        tratti.append(corrente)
    for tratto in tratti:
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                        for i, (x, y) in enumerate(tratto))
        parts.append(f'<path d="{path}" fill="none" stroke="{colore}" '
                     f'stroke-width="2" stroke-linejoin="round"{dash}/>')
        for a, b in zip(tratto, tratto[1:]):
            _campiona(panel, a, b)
    if marcatori:
        for x, y in grezze:
            if not _dentro(panel, x, y):
                continue
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" '
                         f'fill="{colore}" stroke="var(--panel)" '
                         f'stroke-width="2"/>')
            panel.occupati.append((x, y))


def legenda(parts, panel, voci, angolo="auto"):
    """The legend, on a plate of the panel colour, in the corner that covers
    the least data.

    The plate alone is not enough: it hides the curves behind it. In a fixed
    corner it ended up over the thread pipeline, which rises exactly on the
    right. The corner is chosen by counting the points already drawn under each
    of the four, which is why the legend must be called after the series."""
    if not voci:
        return
    larghezza = 8 + 26 + 6 + max(len(t) for _, t, _ in voci) * 6.2 + 8
    altezza = 8 + len(voci) * 15
    angoli = {
        "ne": (panel.x0 + W - larghezza - 8, panel.y0 + 8),
        "nw": (panel.x0 + 8, panel.y0 + 8),
        "sw": (panel.x0 + 8, panel.y0 + H - altezza - 8),
        "se": (panel.x0 + W - larghezza - 8, panel.y0 + H - altezza - 8),
    }
    if angolo == "auto":
        ordine = ("ne", "nw", "sw", "se")
        angolo = min(ordine, key=lambda a: (
            _conta(panel, (angoli[a][0], angoli[a][1], larghezza, altezza)),
            ordine.index(a)))
    x, y = angoli[angolo]
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{larghezza:.1f}" '
                 f'height="{altezza:.1f}" fill="var(--panel)" '
                 f'opacity="0.92"/>')
    for i, (colore, testo, tratteggio) in enumerate(voci):
        yy = y + 14 + i * 15
        dash = f' stroke-dasharray="{tratteggio}"' if tratteggio else ""
        parts.append(f'<line x1="{x + 8:.1f}" y1="{yy:.1f}" '
                     f'x2="{x + 34:.1f}" y2="{yy:.1f}" stroke="{colore}" '
                     f'stroke-width="2"{dash}/>')
        parts.append(f'<text x="{x + 40:.1f}" y="{yy + 4:.1f}" '
                     f'class="legend">{testo}</text>')


def legenda_riga(parts, x, y, voci):
    """The legend on one row, below the panels, once per figure.

    When the curves occupy all four corners no corner is free, and a plate
    inside the panel always covers something."""
    for colore, testo, tratteggio in voci:
        dash = f' stroke-dasharray="{tratteggio}"' if tratteggio else ""
        parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 26:.1f}" '
                     f'y2="{y:.1f}" stroke="{colore}" stroke-width="2"{dash}/>')
        parts.append(f'<text x="{x + 32:.1f}" y="{y + 4:.1f}" '
                     f'class="legend">{testo}</text>')
        x += 32 + len(testo) * 6.2 + 26


def tacche(panel, valori_, formato=lambda v: f"{v:.0f}", minimo=20):
    """The ticks, thinned out: two labels less than `minimo` pixels from each
    other overlap, and on a logarithmic scale that always happens between 7 and
    8 or between 96 and 128."""
    tenute = []
    ultimo = None
    for v in valori_:
        x = panel.px(v)
        if ultimo is None or x - ultimo >= minimo:
            tenute.append(v)
            ultimo = x
    panel.xticks(tenute, [formato(v) for v in tenute])


def griglia_calore(parts, x0, y0, larghezza, altezza, righe, colonne, valori_,
                   etichetta_riga, etichetta_col, titolo, sottotitolo, unita,
                   scala=None):
    """A heat map: darker means slower.

    A single hue from light to dark, because the value has an order and hues do
    not. The empty cells are the configurations that do not fit in the node,
    and they stay empty instead of being drawn as zero."""
    buoni = [v for v in valori_.values() if v is not None]
    if not buoni:
        return
    # The scale comes from outside when there is more than one panel: two maps
    # side by side with different scales invite comparing the colours, and it
    # is the wrong comparison.
    lo, hi = scala if scala else (min(buoni), max(buoni))
    span = math.log10(hi / lo) if hi > lo > 0 else 1.0

    parts.append(f'<text x="{x0}" y="{y0 - 30}" class="title">{titolo}</text>')
    parts.append(f'<text x="{x0}" y="{y0 - 14}" class="subtitle">{sottotitolo}</text>')

    cw = larghezza / max(len(colonne), 1)
    ch = altezza / max(len(righe), 1)
    for i, riga in enumerate(righe):
        for j, col in enumerate(colonne):
            value = valori_.get((riga, col))
            x = x0 + j * cw
            y = y0 + i * ch
            if value is None:
                parts.append(f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" '
                             f'width="{cw - 2:.1f}" height="{ch - 2:.1f}" '
                             f'fill="none" stroke="var(--grid)"/>')
                continue
            f = math.log10(value / lo) / span if span else 0.0
            passo = RAMPA[min(int(f * (len(RAMPA) - 1) + 0.5), len(RAMPA) - 1)]
            # 2px of background between one cell and the next: without it,
            # neighbouring cells merge and the map looks blotchy instead of
            # gridded.
            parts.append(f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" '
                         f'width="{cw - 2:.1f}" height="{ch - 2:.1f}" '
                         f'fill="{passo}"/>')
            chiaro = "#ffffff" if f > 0.55 else "#0b0b0b"
            testo = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
            parts.append(f'<text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 3:.1f}" '
                         f'class="cell" style="fill:{chiaro}">{testo}</text>')
        parts.append(f'<text x="{x0 - 8}" y="{y0 + i * ch + ch / 2 + 3:.1f}" '
                     f'class="tick-y">{etichetta_riga(riga)}</text>')
    for j, col in enumerate(colonne):
        parts.append(f'<text x="{x0 + j * cw + cw / 2:.1f}" '
                     f'y="{y0 + altezza + 15:.1f}" class="tick-x">'
                     f'{etichetta_col(col)}</text>')
    parts.append(f'<text x="{x0 + larghezza / 2:.1f}" '
                 f'y="{y0 + altezza + 38:.1f}" class="axis">{unita}</text>')


def scala_calore(parts, x0, y0, lo, hi, larghezza=180, altezza=10):
    """The map legend: without it, the colours have no unit of measure."""
    passo = larghezza / len(RAMPA)
    for i, colore in enumerate(RAMPA):
        parts.append(f'<rect x="{x0 + i * passo:.1f}" y="{y0}" '
                     f'width="{passo + 0.5:.1f}" height="{altezza}" '
                     f'fill="{colore}"/>')
    parts.append(f'<text x="{x0}" y="{y0 + altezza + 12}" class="tick-x" '
                 f'text-anchor="start">{lo:.3g}</text>')
    parts.append(f'<text x="{x0 + larghezza:.1f}" y="{y0 + altezza + 12}" '
                 f'class="tick-x" text-anchor="end">{hi:.3g}</text>')
    parts.append(f'<text x="{x0 + larghezza / 2:.1f}" y="{y0 - 5}" '
                 f'class="legend" text-anchor="middle">ms per step</text>')


# ---------------------------------------------------------------- the figures


def cubica(row):
    """The rows of phase 11 on a cubic global grid, that is the real case.

    The other two blocks of the phase measure the same question holding the
    local block or its shape fixed: they are rows with different grids from one
    another, and mixing them with these would mean comparing different
    problems."""
    label = row["label"]
    if "bridge" in label or label.startswith(("cube ", "aspect ")):
        return False
    return row["nx"] == row["ny"] == row["nz"]


def fig_asse_puro(rows, outdir):
    """Phase 11, block 2: which axis costs, at constant local block.

    On a cubic grid, cutting z into 56 also means reducing the block to a slab,
    and from the time one cannot tell which of the two things slowed it down.
    Here the local block is the same in every row -- the global grid follows
    the shape -- so the only remaining difference is the axis."""
    data = [r for r in pick(rows, phase="11_matrix_mpi")
            if r["label"].startswith("cube ") and r["ranks"] and r["ranks"] > 1]
    if not data:
        return
    parts = []
    assi = [("x", 0), ("y", 1), ("z", 2)]
    ranks = valori(data, "ranks")
    lato = min(r["nx"] for r in data if r["px"] == 1 and r["py"] == 1) \
        if any(r["px"] == 1 and r["py"] == 1 for r in data) else 0

    panel = Panel(parts, 0, 0, "Which axis really costs",
                  f"same {lato:.0f}^3 local block in every row, "
                  f"only the cuts change",
                  "processes", "ms per step",
                  ranks, [r["wall_ms"] for r in data], xlog=True, ylog=True)
    tacche(panel, ranks)
    voci = []
    for i, (nome, _) in enumerate(assi):
        for k, backend in enumerate(("schur", "pipeline")):
            punti = []
            for n in ranks:
                # The pure cut: all the processes aligned on a single axis.
                gruppo = [r["wall_ms"] for r in pick(data, backend=backend,
                                                     ranks=n)
                          if (r["px"], r["py"], r["pz"]).count(1.0) == 2
                          and (r["px"] if nome == "x" else
                               r["py"] if nome == "y" else r["pz"]) == n]
                if gruppo:
                    punti.append((n, min(gruppo)))
            if not punti:
                continue
            linea(parts, panel, punti, SERIE[i], "" if k == 0 else "6,4",
                  marcatori=(k == 0))
            if k == 0:
                voci.append((SERIE[i], f"{nome} divided", ""))
    voci.append((NEUTRO, "solid: schur", ""))
    voci.append((NEUTRO, "dashed: pipeline", "6,4"))
    legenda(parts, panel, voci)
    # Two rows: a single one would exceed the width of the single panel.
    parts.append(f'<text x="{ML}" y="{MT + H + 56}" class="note">'
                 f'Constant work per process: if the axis did not</text>')
    parts.append(f'<text x="{ML}" y="{MT + H + 72}" class="note">'
                 f'matter, the three curves would lie on top of each other.</text>')
    write(outdir, "matrix-11-asse-puro.svg", 1, 1, parts, serie=3,
          altezza_extra=36)


def fig_aspetto(rows, outdir):
    """Phase 11, block 3: the shape of the local block, at constant volume.

    Same processes, same shape of the process grid, same number of cells per
    process: only whether the block is a cube, a bar or a slab changes. On a
    stencil it is the shape that decides how many cache lines are reused, and
    this is the other half of what the cubic grid kept tied to the axis."""
    data = [r for r in pick(rows, phase="11_matrix_mpi", simd=1.0)
            if r["label"].startswith("aspect ")]
    if not data:
        return
    parts = []
    forme = []
    for r in data:
        nome = r["label"].split()[2]
        if nome not in forme:
            forme.append(nome)
    if not forme:
        return

    for col, backend in enumerate(("schur", "pipeline")):
        qui = pick(data, backend=backend)
        valori_ = []
        for nome in forme:
            gruppo = [r["wall_ms"] for r in qui
                      if r["label"].split()[2] == nome]
            valori_.append(min(gruppo) if gruppo else 0.0)
        if not any(valori_):
            continue
        # The axis title is put by the figure, lower than usual: the rotated
        # labels of the bars occupy the place where it would be.
        panel = Panel(parts, col, 0, f"Block shape, {backend}",
                      "same cells per process, different proportions",
                      "", "ms per step",
                      [0, len(forme)], valori_, xlog=False, ylog=False)
        # A single series, hence a single colour: colouring every bar
        # differently would say twice what the height already says.
        panel.bars(forme, valori_, [SERIE[0]])
        parts.append(f'<text x="{panel.x0 + W / 2}" y="{panel.y0 + H + 64}" '
                     f'class="axis">local block</text>')
    write(outdir, "matrix-11-aspetto.svg", 2, 1, parts, serie=1)


# The stages, grouped to stay within the five verified colours. The three
# momentum steps stay separate because their asymmetry is half of what the
# campaign measures; the three of the pressure do not, because they move
# together.
COMPOSIZIONE = [
    (("g_ms",), "g, right-hand side"),
    (("eta_solve",), "eta, the system"),
    (("zeta_ms", "u_ms"), "zeta and u"),
    (("psi_ms", "philow_ms", "phihigh_ms", "pressure_ms"), "pressure"),
    (("porosity_ms", "untimed_ms"), "porosity and unaccounted"),
]


def fig_composizione(rows, outdir):
    """Where the time of a step goes, stage by stage.

    The CSV times every section separately, and this is the figure that uses
    those columns: stacked bars in milliseconds, not in percent, so that the
    composition and the total are read together. A percentage alone hides that
    one configuration is three times slower than the one above."""
    data = pick(rows, phase="12_matrix_hybrid", simd=1.0)
    # The measurements taken before the binary timed g have the column empty.
    # Here it is not enough to read it as zero: the `g' slice would vanish from
    # the bar and look like a result instead of a missing datum.
    data = [r for r in data if r["g_ms"] is not None]
    if not data:
        return
    grid = max(valori(data, "nx"))
    data = pick(data, nx=grid)

    # A few representative placements, not all: the figure is there to show how
    # the composition changes, not to list the rectangle.
    voluti = [(1, 1), (1, 28), (1, 56), (8, 7), (28, 2), (56, 1)]
    barre = []
    negativi = False
    for backend in ("schur", "pipeline"):
        for r, t in voluti:
            gruppo = pick(data, backend=backend, ranks=float(r),
                          threads=float(t))
            if not gruppo:
                continue
            migliore = min(gruppo, key=lambda g: g["wall_ms"])
            # The `eta, the system' slice is not a column: it is eta minus g,
            # because g sits inside eta and not beside it.
            migliore = dict(migliore)
            migliore["eta_solve"] = max((migliore["eta_ms"] or 0.0)
                                        - (migliore["g_ms"] or 0.0), 0.0)
            pezzi = []
            for chiavi, nome in COMPOSIZIONE:
                valore = sum(migliore[k] or 0.0 for k in chiavi)
                # With several processes every stage is the maximum over the
                # processes, and the sum of the maxima can exceed the step: the
                # unaccounted comes out negative. A negative piece is not
                # drawn, and it counts as zero also for the share, which would
                # otherwise exceed one hundred.
                if valore < 0:
                    negativi = True
                pezzi.append(max(valore, 0.0))
            barre.append((f"{backend}, {piazzamento(r, t)}", pezzi,
                          migliore["wall_ms"]))
    if not barre:
        return

    massimo = max(max(b[2], sum(b[1])) for b in barre) or 1.0
    # Two columns of bars: on the left the milliseconds with a single scale, on
    # the right the same row normalised to one hundred. The first says how much
    # it costs, the second what it is made of -- and with a factor of thirty
    # between the slowest and the fastest row, the first alone does not show
    # the composition.
    larghezza_area = 330
    larghezza_quota = 200
    sinistra = 190
    riga = 26
    parts = []
    x0 = ML + sinistra
    y0 = MT

    parts.append(f'<text x="{ML}" y="{y0 - 26}" class="title">'
                 f'Where the time of a step goes, {grid:.0f}^3</text>')
    parts.append(f'<text x="{ML}" y="{y0 - 10}" class="subtitle">'
                 f'milliseconds on the left, the same row as a '
                 f'percentage share on the right</text>')

    for i, (nome, pezzi, totale) in enumerate(barre):
        y = y0 + i * riga
        parts.append(f'<text x="{x0 - 10}" y="{y + 17:.1f}" '
                     f'class="tick-y">{nome}</text>')
        x = x0
        for j, valore in enumerate(pezzi):
            larghezza = larghezza_area * valore / massimo
            if larghezza <= 0:
                continue
            # 2 pixels of background between one segment and the next:
            # attached, two close hues look like one.
            parts.append(f'<rect x="{x:.1f}" y="{y + 4:.1f}" '
                         f'width="{max(larghezza - 2, 0.5):.1f}" height="16" '
                         f'fill="{SERIE[j]}"/>')
            if larghezza > 34:
                parts.append(f'<text x="{x + larghezza / 2:.1f}" '
                             f'y="{y + 16:.1f}" class="cell" '
                             f'style="fill:#ffffff">{valore:.0f}</text>')
            x += larghezza
        parts.append(f'<text x="{x + 8:.1f}" y="{y + 17:.1f}" class="value" '
                     f'text-anchor="start">{totale:.0f}</text>')

        # The same row as a percentage share, on the right.
        somma = sum(pezzi) or 1.0
        xq = x0 + larghezza_area + 72
        for j, valore in enumerate(pezzi):
            larghezza = larghezza_quota * valore / somma
            if larghezza <= 0:
                continue
            parts.append(f'<rect x="{xq:.1f}" y="{y + 4:.1f}" '
                         f'width="{max(larghezza - 2, 0.5):.1f}" height="16" '
                         f'fill="{SERIE[j]}"/>')
            if larghezza > 26:
                parts.append(f'<text x="{xq + larghezza / 2:.1f}" '
                             f'y="{y + 16:.1f}" class="cell" '
                             f'style="fill:#ffffff">'
                             f'{100 * valore / somma:.0f}</text>')
            xq += larghezza

    # The legend below, in a row: five entries do not fit in a corner.
    y = y0 + len(barre) * riga + 22
    x = x0
    for j, (_, nome) in enumerate(COMPOSIZIONE):
        parts.append(f'<rect x="{x:.1f}" y="{y - 9}" width="12" height="12" '
                     f'fill="{SERIE[j]}"/>')
        parts.append(f'<text x="{x + 18:.1f}" y="{y + 1}" class="legend">'
                     f'{nome}</text>')
        x += 24 + len(nome) * 6.4
    parts.append(f'<text x="{ML}" y="{y + 26}" class="note">'
                 f'The number at the end of each bar is the whole step. '
                 f'The eta column carries the physical term g, the other two do not, '
                 f'so they differ.</text>')
    if negativi:
        parts.append(f'<text x="{ML}" y="{y + 42}" class="note">'
                     f'With several processes every stage is the maximum over the '
                     f'processes: where the sum exceeds the step, the unaccounted '
                     f'is zero.</text>')

    larghezza_svg = ML + sinistra + larghezza_area + 72 + larghezza_quota + 40
    altezza_svg = y + (64 if negativi else 48)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "matrix-12-composizione.svg"
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{larghezza_svg}" '
        f'height="{altezza_svg}" viewBox="0 0 {larghezza_svg} {altezza_svg}">\n'
        f'<style>{STILE}</style>\n'
        f'<rect width="{larghezza_svg}" height="{altezza_svg}" '
        f'fill="var(--background)"/>\n' + "\n".join(parts) + "\n</svg>\n",
        encoding="utf-8")
    print(f"  {path}")


def fig_thread(rows, outdir):
    """Phase 10: one rank, increasing threads. Four series, backend x SIMD."""
    data = [r for r in pick(rows, phase="10_matrix_threads")
            if "bind=" not in r["label"] and "omp=0" not in r["label"]]
    if not data:
        return
    grids = valori(data, "nx")
    parts = []
    voci_legenda = []
    # Two dimensions, two channels: the hue says the backend, the dashing says
    # whether SIMD is on. Four hues would say the same thing with twice the
    # colours, and two of them would be close.
    for col, grid in enumerate(grids):
        here = pick(data, nx=grid)
        panel = Panel(parts, col, 0, f"Threads, {grid:.0f}^3",
                      "one process only, no axis divided",
                      "threads", "ms per step",
                      [r["threads"] for r in here],
                      [r["wall_ms"] for r in here], xlog=True, ylog=True)
        tacche(panel, valori(here, "threads"))
        voci = []
        etichette = []
        for i, backend in enumerate(("schur", "pipeline")):
            for simd, tratto in ((1.0, ""), (0.0, "6,4")):
                punti = sorted((r["threads"], r["wall_ms"])
                               for r in pick(here, backend=backend, simd=simd))
                if not punti:
                    continue
                linea(parts, panel, punti, SERIE[i], tratto,
                      marcatori=(simd == 1.0))
                if simd == 1.0:
                    voci.append((SERIE[i], backend, ""))
                    etichette.append((punti, SERIE[i], backend))
        voci.append((NEUTRO, "dashed: without SIMD", "6,4"))
        base = sorted((r["threads"], r["wall_ms"])
                      for r in pick(here, backend="schur", simd=1.0))
        if base:
            t0, y0 = base[0]
            linea(parts, panel, [(t, y0 * t0 / t) for t, _ in base], NEUTRO,
                  marcatori=False)
            voci.append((NEUTRO, "ideal", ""))
        # The labels after all the lines: each one chooses its place by looking
        # at what is already drawn.
        for punti, colore, testo in etichette:
            etichetta_fine(parts, panel, punti, colore, testo)
        voci_legenda = voci or voci_legenda
    # The legend below, a single one: the pipeline rises on the right, Schur
    # and the ideal descend from the left, and at 224^3 no corner of the panel
    # remains free.
    legenda_riga(parts, ML, MT + H + MB + 8, voci_legenda)
    write(outdir, "matrix-10-thread.svg", len(grids), 1, parts, serie=2,
          altezza_extra=24)


def fig_forme(rows, outdir):
    """Phase 11: how much the time changes depending on HOW it is divided.

    For each number of ranks all the triples (px, py, pz) are measured. Here
    the interval between the best and the worst shape is drawn: it is the cost
    of getting the division wrong, and it is the question of the phase."""
    data = [r for r in pick(rows, phase="11_matrix_mpi")
            if cubica(r)]
    if not data:
        return
    grids = valori(data, "nx")
    parts = []

    for col, grid in enumerate(grids):
        here = pick(data, nx=grid, simd=1.0)
        if not here:
            continue
        ranks = valori(here, "ranks")
        panel = Panel(parts, col, 0, f"Grid shape, {grid:.0f}^3",
                      "interval between the best and the worst shape",
                      "processes", "ms per step",
                      ranks, [r["wall_ms"] for r in here],
                      xlog=True, ylog=True)
        tacche(panel, ranks)
        voci = []
        for i, backend in enumerate(("schur", "pipeline")):
            colore = SERIE[i]
            migliori = []
            for n in ranks:
                gruppo = [r["wall_ms"] for r in pick(here, backend=backend,
                                                     ranks=n)]
                if not gruppo:
                    continue
                lo, hi = min(gruppo), max(gruppo)
                # The lateral offset separates the two backends: without it,
                # the two intervals overlap on the same x and cannot be read.
                x = panel.px(n) + (i - 0.5) * 7
                parts.append(f'<line x1="{x:.1f}" y1="{panel.py(lo):.1f}" '
                             f'x2="{x:.1f}" y2="{panel.py(hi):.1f}" '
                             f'stroke="{colore}" stroke-width="2" '
                             f'stroke-linecap="round"/>')
                parts.append(f'<circle cx="{x:.1f}" cy="{panel.py(lo):.1f}" '
                             f'r="4" fill="{colore}"/>')
                parts.append(f'<circle cx="{x:.1f}" cy="{panel.py(hi):.1f}" '
                             f'r="4" fill="var(--panel)" stroke="{colore}" '
                             f'stroke-width="2"/>')
                _campiona(panel, (x, panel.py(lo)), (x, panel.py(hi)))
                migliori.append((n, lo))
            if migliori:
                linea(parts, panel, migliori, colore, marcatori=False)
                voci.append((colore, backend, ""))
        legenda(parts, panel, voci)
        parts.append(f'<text x="{panel.x0}" y="{panel.y0 + H + 56}" '
                     f'class="note">filled: the best shape. empty: the '
                     f'worst.</text>')
    write(outdir, "matrix-11-forme.svg", len(grids), 1, parts)


def fig_asse_diviso(rows, outdir):
    """Phase 11, the second question: which axis is worth dividing.

    For each backend, the mean time of the shapes that divide a given axis. The
    prediction from the code is that the two backends give different answers,
    and this figure is where it shows or does not show."""
    data = [r for r in pick(rows, phase="11_matrix_mpi", simd=1.0)
            if cubica(r)]
    data = [r for r in data if r["ranks"] and r["ranks"] > 1]
    if not data:
        return
    grids = valori(data, "nx")
    parts = []
    assi = [("px", "x divided"), ("py", "y divided"), ("pz", "z divided")]

    for col, grid in enumerate(grids):
        here = pick(data, nx=grid)
        ranks = valori(here, "ranks")
        panel = Panel(parts, col, 0, f"Which axis to divide, {grid:.0f}^3",
                      "median of the shapes that divide that axis",
                      "processes", "ms per step",
                      ranks, [r["wall_ms"] for r in here],
                      xlog=True, ylog=True)
        tacche(panel, ranks)
        voci = []
        for i, (chiave, nome) in enumerate(assi):
            for k, backend in enumerate(("schur", "pipeline")):
                punti = []
                for n in ranks:
                    gruppo = sorted(r["wall_ms"] for r in
                                    pick(here, backend=backend, ranks=n)
                                    if r[chiave] and r[chiave] > 1)
                    if gruppo:
                        punti.append((n, gruppo[len(gruppo) // 2]))
                if not punti:
                    continue
                # The three series are the axes; the backend is the dashing, so
                # identity is not entrusted to colour twice.
                linea(parts, panel, punti, SERIE[i],
                      "" if k == 0 else "6,4", marcatori=(k == 0))
                if k == 0:
                    voci.append((SERIE[i], nome, ""))
        voci.append((NEUTRO, "solid: schur", ""))
        voci.append((NEUTRO, "dashed: pipeline", "6,4"))
        legenda(parts, panel, voci)
    write(outdir, "matrix-11-assi.svg", len(grids), 1, parts, serie=3)


def fig_rettangolo(rows, outdir):
    """Phase 12: the full rank x thread rectangle, one map per backend."""
    data = pick(rows, phase="12_matrix_hybrid", simd=1.0)
    if not data:
        return
    grid = max(valori(data, "nx"))
    data = pick(data, nx=grid)
    ranks = valori(data, "ranks")
    threads = valori(data, "threads")
    if not ranks or not threads:
        return

    parts = []
    larghezza, altezza = W + 40, H
    tutti = [r["wall_ms"] for r in data if r["wall_ms"]]
    scala = (min(tutti), max(tutti))
    for col, backend in enumerate(("schur", "pipeline")):
        x0 = ML + col * (larghezza + ML + MR)
        y0 = MT
        celle = {}
        for r in ranks:
            for t in threads:
                gruppo = pick(data, backend=backend, ranks=r, threads=t)
                celle[(r, t)] = min((g["wall_ms"] for g in gruppo),
                                    default=None)
        griglia_calore(parts, x0, y0, larghezza, altezza, ranks, threads,
                       celle, lambda r: f"{r:.0f}", lambda t: f"{t:.0f}",
                       f"{backend}, {grid:.0f}^3",
                       "ms per step; rows processes, columns threads",
                       "threads per process", scala=scala)
        parts.append(f'<text x="{x0 - 52}" y="{y0 + altezza / 2}" class="axis" '
                     f'transform="rotate(-90 {x0 - 52} {y0 + altezza / 2})">'
                     f'processes</text>')
        scala_calore(parts, x0, y0 + altezza + 66, scala[0], scala[1])
    parts.append(f'<text x="{ML}" y="{MT + altezza + 120}" class="note">'
                 f'Same colour scale in the two panels, so they can be '
                 f'compared.</text>')
    parts.append(f'<text x="{ML}" y="{MT + altezza + 136}" class="note">'
                 f'Along an anti-diagonal processes x threads is constant: '
                 f'only how the same units are divided changes.</text>')
    width_cols = 2
    write(outdir, "matrix-12-rettangolo.svg", width_cols, 1, parts,
          altezza_extra=86)


def fig_batch(rows, outdir):
    """Phase 13: the pipeline batch, one panel per placement.

    One series per panel plus the Schur reference in grey: the question is
    where the minimum of each curve lies, not how they compare with one
    another, and eight overlapping curves would have hidden it."""
    data = pick(rows, phase="13_matrix_batch", simd=1.0)
    if not data:
        return
    grid = max(valori(data, "nx"))
    data = pick(data, nx=grid)
    piazzamenti = sorted({(r["ranks"], r["threads"]) for r in data})
    if not piazzamenti:
        return

    colonne = min(4, len(piazzamenti))
    righe = (len(piazzamenti) + colonne - 1) // colonne
    parts = []
    for k, (r, t) in enumerate(piazzamenti):
        qui = pick(data, ranks=r, threads=t)
        # batch=auto has no abscissa: it sits outside the curve.
        pipe = sorted((g["batch"], g["wall_ms"])
                      for g in pick(qui, backend="pipeline")
                      if g["batch"] is not None)
        schur = [g["wall_ms"] for g in pick(qui, backend="schur")]
        if not pipe:
            continue
        ys = [y for _, y in pipe] + schur
        panel = Panel(parts, k % colonne, k // colonne,
                      piazzamento(r, t),
                      f"{grid:.0f}^3, pipeline batch",
                      "lines per batch", "ms per step",
                      [x for x, _ in pipe], ys, xlog=True, ylog=False)
        tacche(panel, [x for x, _ in pipe])
        linea(parts, panel, pipe, SERIE[0])
        voci = [(SERIE[0], "pipeline", "")]
        if schur:
            y = min(schur)
            parts.append(f'<line x1="{panel.x0}" y1="{panel.py(y):.1f}" '
                         f'x2="{panel.x0 + W}" y2="{panel.py(y):.1f}" '
                         f'stroke="{NEUTRO}" stroke-width="2"/>')
            _campiona(panel, (panel.x0, panel.py(y)), (panel.x0 + W, panel.py(y)))
            voci.append((NEUTRO, "schur, same placement", ""))
        # The minimum, marked: it is the only thing the reader has to take
        # away.
        bx, by = min(pipe, key=lambda p: p[1])
        parts.append(f'<circle cx="{panel.px(bx):.1f}" cy="{panel.py(by):.1f}" '
                     f'r="7" fill="none" stroke="{SERIE[0]}" '
                     f'stroke-width="2"/>')
        parts.append(f'<text x="{panel.px(bx):.1f}" '
                     f'y="{panel.py(by) - 12:.1f}" class="value" '
                     f'style="fill:{SERIE[0]}">{bx:.0f}</text>')
        _occupa(panel, (panel.px(bx) - 14, panel.py(by) - 22, 28, 30))
        legenda(parts, panel, voci)
    parts.append(f'<text x="{ML}" y="{MT + righe * (H + MT + MB) + 10}" '
                 f'class="note">Each panel has its own vertical scale: the '
                 f'question is where the minimum of each curve lies, not how '
                 f'they compare with one another.</text>')
    write(outdir, "matrix-13-batch.svg", colonne, righe, parts, serie=1,
          altezza_extra=24)


def fig_taglia(rows, outdir):
    """Phase 14: the cost per cell as the size grows.

    If the work were only arithmetic the cost per cell would stay flat. Where
    it rises, the local block has left the cache: it is the memory-bandwidth
    wall, and knowing where it lies is what allows reading all the other
    figures."""
    data = [r for r in pick(rows, phase="14_matrix_size")
            if " N=" in r["label"]]
    if not data:
        return
    piazzamenti = sorted({(r["ranks"], r["threads"]) for r in data})
    colonne = min(4, len(piazzamenti))
    righe = (len(piazzamenti) + colonne - 1) // colonne
    parts = []

    for k, (r, t) in enumerate(piazzamenti):
        qui = pick(data, ranks=r, threads=t)
        ys = [g["cellstep_1e8s"] for g in qui if g["cellstep_1e8s"]]
        if not ys:
            continue
        panel = Panel(parts, k % colonne, k // colonne,
                      piazzamento(r, t),
                      "cost per cell and per step",
                      "grid side", "1e-8 s per cell",
                      valori(qui, "nx"), ys, xlog=True, ylog=False)
        tacche(panel, valori(qui, "nx"))
        voci = []
        for i, backend in enumerate(("schur", "pipeline")):
            for simd, tratto in ((1.0, ""), (0.0, "6,4")):
                punti = sorted((g["nx"], g["cellstep_1e8s"])
                               for g in pick(qui, backend=backend, simd=simd)
                               if g["cellstep_1e8s"])
                if not punti:
                    continue
                linea(parts, panel, punti, SERIE[i], tratto,
                      marcatori=(simd == 1.0))
                if simd == 1.0:
                    voci.append((SERIE[i], backend, ""))
        voci.append((NEUTRO, "dashed: without SIMD", "6,4"))
        legenda(parts, panel, voci)
    write(outdir, "matrix-14-taglia.svg", colonne, righe, parts, serie=2)


def fig_memoria(rows, outdir):
    """Phase 14: the peak memory, which for the pipeline is its price.

    It keeps c' and d' of the whole local block for three components: it pays
    in memory what Schur pays in arithmetic. On an axis of its own, because two
    quantities never go on two y scales of the same panel."""
    data = [r for r in pick(rows, phase="14_matrix_size")
            if " N=" in r["label"] and r["rss_mb"]]
    if not data:
        return
    piazzamenti = sorted({(r["ranks"], r["threads"]) for r in data})[:4]
    parts = []
    for k, (r, t) in enumerate(piazzamenti):
        qui = pick(data, ranks=r, threads=t, simd=1.0)
        if not qui:
            continue
        panel = Panel(parts, k, 0, piazzamento(r, t),
                      "peak memory per process",
                      "grid side", "MB",
                      valori(qui, "nx"), [g["rss_mb"] for g in qui],
                      xlog=True, ylog=True)
        tacche(panel, valori(qui, "nx"))
        voci = []
        for i, backend in enumerate(("schur", "pipeline")):
            punti = sorted((g["nx"], g["rss_mb"])
                           for g in pick(qui, backend=backend))
            if not punti:
                continue
            linea(parts, panel, punti, SERIE[i])
            voci.append((SERIE[i], backend, ""))
        legenda(parts, panel, voci)
    write(outdir, "matrix-14-memoria.svg", len(piazzamenti), 1, parts, serie=2)


def fig_scaling(rows, outdir):
    """Phase 14: strong and weak scaling, with the ideal line beside it."""
    forte = [r for r in pick(rows, phase="14_matrix_size")
             if " strong " in r["label"]]
    debole = [r for r in pick(rows, phase="14_matrix_size")
              if " weak " in r["label"]]
    if not forte and not debole:
        return
    parts = []
    voci_legenda = []

    if forte:
        unita = sorted({max(r["ranks"], r["threads"]) for r in forte})
        # The series before the panel: the scale must also contain the speedups
        # below one -- the pipeline on threads goes down there -- or those
        # points stay outside the panel.
        serie_forti = []
        i = 0
        for backend in ("schur", "pipeline"):
            for modo, chiave in (("processes", "ranks"), ("threads", "threads")):
                serie = [r for r in forte if r["backend"] == backend
                         and (r["threads"] == 1 if chiave == "ranks"
                              else r["ranks"] == 1)]
                punti = sorted((r[chiave], r["wall_ms"]) for r in serie)
                if len(punti) < 2:
                    continue
                base = punti[0][1]
                tratto = "" if i % 2 == 0 else "6,4"
                serie_forti.append(([(u, base / w) for u, w in punti],
                                    SERIE[i // 2], tratto,
                                    f"{backend}, {modo}"))
                i += 1
        speedup = [v for punti, _, _, _ in serie_forti for _, v in punti]
        panel = Panel(parts, 0, 0, "Strong scaling",
                      "same problem, more compute units",
                      "units (processes or threads)", "speedup",
                      # Log on both axes: this way the ideal line is a straight
                      # line, and the deviation reads the same at 2 units and
                      # at 56 instead of being squashed at the bottom left.
                      unita, speedup + [1, max(unita)], xlog=True, ylog=True)
        tacche(panel, unita)
        voci = []
        for punti, colore, tratto, nome in serie_forti:
            linea(parts, panel, punti, colore, tratto,
                  marcatori=(tratto == ""))
            voci.append((colore, nome, tratto))
        linea(parts, panel, [(u, u) for u in unita], NEUTRO, marcatori=False)
        voci.append((NEUTRO, "ideal", ""))
        voci_legenda = voci

    if debole:
        ranks = valori(debole, "ranks")
        panel = Panel(parts, 1, 0, "Weak scaling",
                      "constant cells per process",
                      "processes", "efficiency",
                      ranks, [0.0, 1.15], xlog=True, ylog=False)
        tacche(panel, ranks)
        voci = []
        for i, backend in enumerate(("schur", "pipeline")):
            punti = sorted((r["ranks"], r["wall_ms"])
                           for r in pick(debole, backend=backend, simd=1.0))
            if len(punti) < 2:
                continue
            base = punti[0][1]
            linea(parts, panel, [(n, base / w) for n, w in punti], SERIE[i])
            voci.append((SERIE[i], backend, ""))
        linea(parts, panel, [(n, 1.0) for n in ranks], NEUTRO, marcatori=False)
        voci.append((NEUTRO, "ideal", ""))
        voci_legenda = voci_legenda or voci

    # A single legend, below the two panels: in the strong case the five series
    # occupied every corner, and the weak one uses the same hues and the same
    # stroke.
    legenda_riga(parts, ML, MT + H + MB + 8, voci_legenda)
    write(outdir, "matrix-14-scaling.svg", 2, 1, parts, serie=2,
          altezza_extra=24)


def fig_norme(rows, outdir):
    """Phase 15: does the campaign still solve the right problem?

    A single number, and therefore not a plot: how far the worst configuration
    departs from the reference. Expected zero, not `small'."""
    data = [r for r in pick(rows, phase="15_matrix_check") if r["l2_ux"]]
    if not data:
        return
    riferimento = data[0]
    peggiore_ux = 0.0
    peggiore_p = 0.0
    chi = riferimento["label"]
    for row in data:
        if riferimento["l2_ux"]:
            d = abs(row["l2_ux"] - riferimento["l2_ux"]) / abs(riferimento["l2_ux"])
            if d > peggiore_ux:
                peggiore_ux, chi = d, row["label"]
        if riferimento["l2_p"]:
            d = abs(row["l2_p"] - riferimento["l2_p"]) / abs(riferimento["l2_p"])
            peggiore_p = max(peggiore_p, d)

    peggiore = max(peggiore_ux, peggiore_p)
    colore = SERIE[2] if peggiore == 0 else SERIE[1]
    esito = ("all the configurations give the same answer"
             if peggiore == 0 else
             f"one configuration departs from it: {chi}")
    parts = [
        f'<text x="{ML}" y="{MT - 14}" class="title">'
        f'Correctness over the whole matrix</text>',
        f'<text x="{ML}" y="{MT + 6}" class="subtitle">'
        f'{len(data)} configurations compared with {riferimento["label"]}'
        f'</text>',
        f'<text x="{ML}" y="{MT + 74}" class="huge" style="fill:{colore}">'
        f'{peggiore:.2e}</text>',
        f'<text x="{ML}" y="{MT + 98}" class="note">'
        f'maximum relative deviation of the L2 norms</text>',
        f'<text x="{ML}" y="{MT + 122}" class="note">{esito}</text>',
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "matrix-15-norme.svg"
    width, height = 640, 200
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        f'<style>{STILE}</style>\n'
        f'<rect width="{width}" height="{height}" fill="var(--background)"/>\n'
        + "\n".join(parts) + "\n</svg>\n", encoding="utf-8")
    print(f"  {path}")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", type=Path,
                        default=root / "build" / "study" / "all.csv",
                        help="the CSV merged by ./scripts/run_study.sh merge")
    parser.add_argument("-o", "--outdir", type=Path,
                        default=root / "docs" / "scaling" / "matrix",
                        help="where to write the SVGs")
    args = parser.parse_args()

    rows = load(args.csv)
    fig_thread(rows, args.outdir)
    fig_forme(rows, args.outdir)
    fig_asse_diviso(rows, args.outdir)
    fig_asse_puro(rows, args.outdir)
    fig_aspetto(rows, args.outdir)
    fig_rettangolo(rows, args.outdir)
    fig_composizione(rows, args.outdir)
    fig_batch(rows, args.outdir)
    fig_taglia(rows, args.outdir)
    fig_memoria(rows, args.outdir)
    fig_scaling(rows, args.outdir)
    fig_norme(rows, args.outdir)


if __name__ == "__main__":
    main()
