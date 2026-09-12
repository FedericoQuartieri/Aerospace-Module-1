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

# ------------------------------------------------------------- i riquadri
#
# Geometria dei pannelli: margini, tacche, scale logaritmiche, barre. Stava in
# plot_study.py, lo script dello studio a dieci fasi; quello e' stato tolto
# insieme alle fasi che disegnava, e la classe si e' trasferita qui.

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

# --------------------------------------------------------------------- colori
#
# Scala categorica, nell'ordine: e' l'ordine a garantire la separazione fra
# tinte vicine, quindi le serie si assegnano dalla prima in poi e non si
# rimescolano quando un filtro ne toglie una. Una nona serie non esiste: si
# accorpa o si divide la figura.

SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SERIE_SCURO = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
NEUTRO = "#8a8a83"

# Scala sequenziale per le mappe di calore: una tinta sola, dal chiaro allo
# scuro. Mai un arcobaleno: le tinte non hanno un ordine naturale, i valori si.
RAMPA = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
         "#0d366b"]

STILE = """
svg{--fondo:#fcfcfb;--inchiostro:#0b0b0b;--inchiostro-2:#52514e;
    --inchiostro-3:#6f6e69;--riquadro:#f6f6f4;--linea:#d9d8d2;--griglia:#e8e7e2}
@media (prefers-color-scheme:dark){
svg{--fondo:#1a1a19;--inchiostro:#ffffff;--inchiostro-2:#c3c2b7;
    --inchiostro-3:#a3a299;--riquadro:#232322;--linea:#3a3a37;--griglia:#2e2e2c}}
text{font-family:"DejaVu Sans",sans-serif;fill:var(--inchiostro)}
.titolo{font-size:15px;font-weight:600}
.sotto{font-size:11px;fill:var(--inchiostro-2)}
.asse{font-size:12px;fill:var(--inchiostro-2);text-anchor:middle}
.tacca-x{font-size:10px;fill:var(--inchiostro-3);text-anchor:middle}
.tacca-y{font-size:10px;fill:var(--inchiostro-3);text-anchor:end}
.legenda{font-size:11px;fill:var(--inchiostro-2)}
.valore{font-size:9px;fill:var(--inchiostro-2);text-anchor:middle}
.cella{font-size:9px;text-anchor:middle}
.nota{font-size:11px;fill:var(--inchiostro-2)}
.enorme{font-size:46px;font-weight:600}
.riquadro{fill:var(--riquadro);stroke:var(--linea)}
.griglia{stroke:var(--griglia);stroke-width:1}
"""

# Le tinte scure si applicano riscrivendo la variabile: un blocco per ogni
# serie usata, generato quando serve.


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
        f'<rect width="{width}" height="{height}" fill="var(--fondo)"/>',
    ]
    return "\n".join(head + parts + ["</svg>"])


def write(outdir, name, cols, rows, parts, serie=4, altezza_extra=0):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    path.write_text(svg(cols, rows, parts, serie, altezza_extra),
                    encoding="utf-8")
    print(f"  {path}")


# ----------------------------------------------------------------- i dati


NUMERICHE = ("batch", "simd", "omp", "mpi", "ranks", "threads", "nx", "ny",
             "nz", "steps", "px", "py", "pz", "wall_ms", "mpi_ms", "eta_ms",
             "zeta_ms", "u_ms", "psi_ms", "philow_ms", "phihigh_ms",
             "pressure_ms", "porosity_ms", "untimed_ms", "cellstep_1e8s",
             "rss_mb", "l2_ux", "l2_p", "g_ms")


def load(path):
    """Le righe riuscite, con i numeri gia' convertiti.

    Un caso fallito o in timeout porta colonne vuote: tenerlo vorrebbe dire
    disegnare uno zero dove non c'e' una misura, ed e' il modo piu' rapido di
    leggere un buco come un risultato."""
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
            # Le due righe di riferimento di ogni configurazione si
            # riconoscono dal suffisso che study_baseline mette in coda
            # all'etichetta. Marcarle qui costa una volta sola e permette a
            # pick() di tenerle fuori dalle curve.
            label = row.get("label") or ""
            if label.endswith(" seriale"):
                row["baseline"] = "seriale"
            elif label.endswith(" T(1)"):
                row["baseline"] = "T(1)"
            else:
                row["baseline"] = None
            rows.append(row)
    print(f"{len(rows)} valid measurements from {path}")
    return rows


def pick(rows, **filtri):
    """Le righe che soddisfano i filtri, riferimenti esclusi.

    Seriale e T(1) non sono punti di una curva, sono i denominatori: hanno un
    rank e un thread, quindi senza questo finirebbero dentro ogni grafico come
    se fossero il caso a un processo -- che pero' c'e' gia' ed e' un altro.
    Per averli si chiede baseline="seriale" o baseline="T(1)"."""
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


def etichetta_fine(parts, panel, points, colore, testo):
    """L'etichetta accanto all'ultimo punto della linea.

    Due delle quattro tinte non arrivano a 3:1 di contrasto sul fondo chiaro:
    la legenda da sola non basta a distinguerle, e l'etichetta attaccata alla
    serie e' quello che rende la figura leggibile anche stampata in bianco e
    nero."""
    if not points:
        return
    x, y = points[-1]
    # Dentro il riquadro, sopra l'ultimo punto: fuori finirebbe addosso
    # all'asse del riquadro accanto, che e' a 26 pixel.
    parts.append(f'<text x="{panel.px(x) - 6:.1f}" y="{panel.py(y) - 10:.1f}" '
                 f'class="legenda" text-anchor="end" '
                 f'style="fill:{colore}">{testo}</text>')


def piazzamento(r, t):
    """`1 processo x 1 thread', non `1 processi'."""
    return (f"{r:.0f} process{'o' if r == 1 else 'i'} x "
            f"{t:.0f} thread")


def linea(parts, panel, punti, colore, tratteggio="", marcatori=True):
    """Una serie. Tratto sottile, marcatori da 9px: sotto quella misura un
    punto si perde, sopra i marcatori diventano il grafico."""
    punti = [p for p in punti if p[1] is not None]
    if not punti:
        return
    coords = [(panel.px(x), panel.py(y)) for x, y in punti]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                    for i, (x, y) in enumerate(coords))
    dash = f' stroke-dasharray="{tratteggio}"' if tratteggio else ""
    parts.append(f'<path d="{path}" fill="none" stroke="{colore}" '
                 f'stroke-width="2" stroke-linejoin="round"{dash}/>')
    if marcatori:
        for x, y in coords:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" '
                         f'fill="{colore}" stroke="var(--riquadro)" '
                         f'stroke-width="2"/>')


def legenda(parts, panel, voci, angolo="ne"):
    """La legenda, su un piatto del colore del riquadro.

    Senza il piatto finisce sopra le curve, ed e' quello che succede sempre
    quando i dati scendono da sinistra a destra."""
    if not voci:
        return
    larghezza = 8 + 26 + 6 + max(len(t) for _, t, _ in voci) * 6.2 + 8
    altezza = 8 + len(voci) * 15
    x = panel.x0 + W - larghezza - 8 if angolo.endswith("e") else panel.x0 + 8
    y = panel.y0 + 8
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{larghezza:.1f}" '
                 f'height="{altezza:.1f}" fill="var(--riquadro)" '
                 f'opacity="0.92"/>')
    for i, (colore, testo, tratteggio) in enumerate(voci):
        yy = y + 14 + i * 15
        dash = f' stroke-dasharray="{tratteggio}"' if tratteggio else ""
        parts.append(f'<line x1="{x + 8:.1f}" y1="{yy:.1f}" '
                     f'x2="{x + 34:.1f}" y2="{yy:.1f}" stroke="{colore}" '
                     f'stroke-width="2"{dash}/>')
        parts.append(f'<text x="{x + 40:.1f}" y="{yy + 4:.1f}" '
                     f'class="legenda">{testo}</text>')


def tacche(panel, valori_, formato=lambda v: f"{v:.0f}", minimo=20):
    """Le tacche, diradate: due etichette a meno di `minimo` pixel l'una
    dall'altra si sovrappongono, e su scala logaritmica capita sempre fra 7 e
    8 o fra 96 e 128."""
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
    """Una mappa di calore: piu' scuro vuol dire piu' lento.

    Una tinta sola dal chiaro allo scuro, perche' il valore ha un ordine e le
    tinte no. Le celle vuote sono le configurazioni che non stanno nel nodo, e
    restano vuote invece di essere disegnate a zero."""
    buoni = [v for v in valori_.values() if v is not None]
    if not buoni:
        return
    # La scala arriva da fuori quando i riquadri sono piu' di uno: due mappe
    # affiancate con scale diverse invitano a confrontare i colori, ed e' il
    # confronto sbagliato.
    lo, hi = scala if scala else (min(buoni), max(buoni))
    span = math.log10(hi / lo) if hi > lo > 0 else 1.0

    parts.append(f'<text x="{x0}" y="{y0 - 30}" class="titolo">{titolo}</text>')
    parts.append(f'<text x="{x0}" y="{y0 - 14}" class="sotto">{sottotitolo}</text>')

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
                             f'fill="none" stroke="var(--griglia)"/>')
                continue
            f = math.log10(value / lo) / span if span else 0.0
            passo = RAMPA[min(int(f * (len(RAMPA) - 1) + 0.5), len(RAMPA) - 1)]
            # 2px di fondo fra una cella e l'altra: senza, le celle vicine si
            # fondono e la mappa sembra a macchie invece che a griglia.
            parts.append(f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" '
                         f'width="{cw - 2:.1f}" height="{ch - 2:.1f}" '
                         f'fill="{passo}"/>')
            chiaro = "#ffffff" if f > 0.55 else "#0b0b0b"
            testo = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
            parts.append(f'<text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 3:.1f}" '
                         f'class="cella" style="fill:{chiaro}">{testo}</text>')
        parts.append(f'<text x="{x0 - 8}" y="{y0 + i * ch + ch / 2 + 3:.1f}" '
                     f'class="tacca-y">{etichetta_riga(riga)}</text>')
    for j, col in enumerate(colonne):
        parts.append(f'<text x="{x0 + j * cw + cw / 2:.1f}" '
                     f'y="{y0 + altezza + 15:.1f}" class="tacca-x">'
                     f'{etichetta_col(col)}</text>')
    parts.append(f'<text x="{x0 + larghezza / 2:.1f}" '
                 f'y="{y0 + altezza + 38:.1f}" class="asse">{unita}</text>')


def scala_calore(parts, x0, y0, lo, hi, larghezza=180, altezza=10):
    """La legenda della mappa: senza, i colori non hanno unita' di misura."""
    passo = larghezza / len(RAMPA)
    for i, colore in enumerate(RAMPA):
        parts.append(f'<rect x="{x0 + i * passo:.1f}" y="{y0}" '
                     f'width="{passo + 0.5:.1f}" height="{altezza}" '
                     f'fill="{colore}"/>')
    parts.append(f'<text x="{x0}" y="{y0 + altezza + 12}" class="tacca-x" '
                 f'text-anchor="start">{lo:.3g}</text>')
    parts.append(f'<text x="{x0 + larghezza:.1f}" y="{y0 + altezza + 12}" '
                 f'class="tacca-x" text-anchor="end">{hi:.3g}</text>')
    parts.append(f'<text x="{x0 + larghezza / 2:.1f}" y="{y0 - 5}" '
                 f'class="legenda" text-anchor="middle">ms per passo</text>')


# ------------------------------------------------------------------ le figure


def cubica(row):
    """Le righe della fase 11 a griglia globale cubica, cioe' il caso reale.

    Gli altri due blocchi della fase misurano la stessa domanda tenendo fisso
    il blocco locale o la sua forma: sono righe con griglie diverse fra loro,
    e mescolarle a queste vorrebbe dire confrontare problemi diversi."""
    label = row["label"]
    if "ponte" in label or label.startswith(("cubo ", "aspetto ")):
        return False
    return row["nx"] == row["ny"] == row["nz"]


def fig_asse_puro(rows, outdir):
    """Fase 11, blocco 2: quale asse costa, a blocco locale costante.

    Su griglia cubica tagliare z in 56 significa anche ridurre il blocco a una
    lamina, e dal tempo non si distingue quale delle due cose l'ha rallentato.
    Qui il blocco locale e' lo stesso in ogni riga -- la griglia globale segue
    la forma -- quindi l'unica differenza rimasta e' l'asse."""
    data = [r for r in pick(rows, phase="11_matrix_mpi")
            if r["label"].startswith("cubo ") and r["ranks"] and r["ranks"] > 1]
    if not data:
        return
    parts = []
    assi = [("x", 0), ("y", 1), ("z", 2)]
    ranks = valori(data, "ranks")
    lato = min(r["nx"] for r in data if r["px"] == 1 and r["py"] == 1) \
        if any(r["px"] == 1 and r["py"] == 1 for r in data) else 0

    panel = Panel(parts, 0, 0, "Quale asse costa davvero",
                  f"blocco locale {lato:.0f}^3 uguale in ogni riga, "
                  f"solo i tagli cambiano",
                  "processi", "ms per passo",
                  ranks, [r["wall_ms"] for r in data], xlog=True, ylog=True)
    tacche(panel, ranks)
    voci = []
    for i, (nome, _) in enumerate(assi):
        for k, backend in enumerate(("schur", "pipeline")):
            punti = []
            for n in ranks:
                # Il taglio puro: tutti i processi allineati su un asse solo.
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
                voci.append((SERIE[i], f"{nome} diviso", ""))
    voci.append((NEUTRO, "continuo: schur", ""))
    voci.append((NEUTRO, "tratteggio: pipeline", "6,4"))
    legenda(parts, panel, voci, angolo="nw")
    # Due righe: una sola sfonderebbe la larghezza del riquadro singolo.
    parts.append(f'<text x="{ML}" y="{MT + H + 56}" class="nota">'
                 f'Lavoro per processo costante: se l\'asse non</text>')
    parts.append(f'<text x="{ML}" y="{MT + H + 72}" class="nota">'
                 f'contasse, le tre curve starebbero una sull\'altra.</text>')
    write(outdir, "matrix-11-asse-puro.svg", 1, 1, parts, serie=3,
          altezza_extra=36)


def fig_aspetto(rows, outdir):
    """Fase 11, blocco 3: la forma del blocco locale, a volume costante.

    Stessi processi, stessa forma della griglia di processi, stesso numero di
    celle per processo: cambia solo se il blocco e' un cubo, una barra o una
    lamina. Su uno stencil e' la forma a decidere quante linee di cache si
    riusano, e questa e' l'altra meta' di quello che la griglia cubica teneva
    insieme all'asse."""
    data = [r for r in pick(rows, phase="11_matrix_mpi", simd=1.0)
            if r["label"].startswith("aspetto ")]
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
        panel = Panel(parts, col, 0, f"Forma del blocco, {backend}",
                      "stesse celle per processo, proporzioni diverse",
                      "blocco locale", "ms per passo",
                      [0, len(forme)], valori_, xlog=False, ylog=False)
        # Una serie sola, quindi un colore solo: colorare ogni barra
        # diversamente direbbe due volte quello che dice gia' l'altezza.
        panel.bars(forme, valori_, [SERIE[0]])
    write(outdir, "matrix-11-aspetto.svg", 2, 1, parts, serie=1)


# Gli stadi, raggruppati per stare dentro i cinque colori verificati. I tre
# passi della quantita' di moto restano separati perche' la loro asimmetria e'
# meta' di quello che la campagna misura; i tre della pressione no, perche' si
# muovono insieme.
COMPOSIZIONE = [
    (("g_ms",), "g, termine noto"),
    (("eta_solve",), "eta, il sistema"),
    (("zeta_ms", "u_ms"), "zeta e u"),
    (("psi_ms", "philow_ms", "phihigh_ms", "pressure_ms"), "pressione"),
    (("porosity_ms", "untimed_ms"), "porosita\' e non contato"),
]


def fig_composizione(rows, outdir):
    """Dove va il tempo di un passo, stadio per stadio.

    Il CSV cronometra ogni sezione separatamente, e questa e' la figura che
    usa quelle colonne: barre impilate in millisecondi, non in percentuale,
    cosi' si legge insieme la composizione e il totale. Una percentuale da
    sola nasconde che una configurazione e' tre volte piu' lenta di quella
    sopra."""
    data = pick(rows, phase="12_matrix_hybrid", simd=1.0)
    # Le misure prese prima che il binario cronometrasse g hanno la colonna
    # vuota. Qui non basta leggerla come zero: la fetta `g' sparirebbe dalla
    # barra e sembrerebbe un risultato invece che un dato mancante.
    data = [r for r in data if r["g_ms"] is not None]
    if not data:
        return
    grid = max(valori(data, "nx"))
    data = pick(data, nx=grid)

    # Qualche piazzamento rappresentativo, non tutti: la figura serve a
    # vedere come cambia la composizione, non a elencare il rettangolo.
    voluti = [(1, 1), (1, 28), (1, 56), (8, 7), (28, 2), (56, 1)]
    barre = []
    for backend in ("schur", "pipeline"):
        for r, t in voluti:
            gruppo = pick(data, backend=backend, ranks=float(r),
                          threads=float(t))
            if not gruppo:
                continue
            migliore = min(gruppo, key=lambda g: g["wall_ms"])
            # `eta il sistema' non e' una colonna: e' eta meno g, perche' g
            # sta dentro eta e non accanto.
            migliore = dict(migliore)
            migliore["eta_solve"] = max((migliore["eta_ms"] or 0.0)
                                        - (migliore["g_ms"] or 0.0), 0.0)
            pezzi = []
            for chiavi, nome in COMPOSIZIONE:
                pezzi.append(sum(migliore[k] or 0.0 for k in chiavi))
            barre.append((f"{backend}, {piazzamento(r, t)}", pezzi,
                          migliore["wall_ms"]))
    if not barre:
        return

    massimo = max(b[2] for b in barre) or 1.0
    # Due colonne di barre: a sinistra i millisecondi con una scala sola, a
    # destra la stessa riga normalizzata a cento. La prima dice quanto costa,
    # la seconda com'e' fatta -- e con trenta volte fra la riga piu' lenta e
    # la piu' veloce, la prima da sola non fa vedere la composizione.
    larghezza_area = 330
    larghezza_quota = 200
    sinistra = 190
    riga = 26
    parts = []
    x0 = ML + sinistra
    y0 = MT

    parts.append(f'<text x="{ML}" y="{y0 - 26}" class="titolo">'
                 f'Dove va il tempo di un passo, {grid:.0f}^3</text>')
    parts.append(f'<text x="{ML}" y="{y0 - 10}" class="sotto">'
                 f'a sinistra i millisecondi, a destra la stessa riga in '
                 f'quota percentuale</text>')

    for i, (nome, pezzi, totale) in enumerate(barre):
        y = y0 + i * riga
        parts.append(f'<text x="{x0 - 10}" y="{y + 17:.1f}" '
                     f'class="tacca-y">{nome}</text>')
        x = x0
        for j, valore in enumerate(pezzi):
            larghezza = larghezza_area * valore / massimo
            if larghezza <= 0:
                continue
            # 2 pixel di fondo fra un segmento e l'altro: attaccati, due
            # tinte vicine sembrano una sola.
            parts.append(f'<rect x="{x:.1f}" y="{y + 4:.1f}" '
                         f'width="{max(larghezza - 2, 0.5):.1f}" height="16" '
                         f'fill="{SERIE[j]}"/>')
            if larghezza > 34:
                parts.append(f'<text x="{x + larghezza / 2:.1f}" '
                             f'y="{y + 16:.1f}" class="cella" '
                             f'style="fill:#ffffff">{valore:.0f}</text>')
            x += larghezza
        parts.append(f'<text x="{x + 8:.1f}" y="{y + 17:.1f}" class="valore" '
                     f'text-anchor="start">{totale:.0f}</text>')

        # La stessa riga in quota percentuale, a destra.
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
                             f'y="{y + 16:.1f}" class="cella" '
                             f'style="fill:#ffffff">'
                             f'{100 * valore / somma:.0f}</text>')
            xq += larghezza

    # La legenda sotto, in riga: cinque voci non stanno in un angolo.
    y = y0 + len(barre) * riga + 22
    x = x0
    for j, (_, nome) in enumerate(COMPOSIZIONE):
        parts.append(f'<rect x="{x:.1f}" y="{y - 9}" width="12" height="12" '
                     f'fill="{SERIE[j]}"/>')
        parts.append(f'<text x="{x + 18:.1f}" y="{y + 1}" class="legenda">'
                     f'{nome}</text>')
        x += 24 + len(nome) * 6.4
    parts.append(f'<text x="{ML}" y="{y + 26}" class="nota">'
                 f'Il numero in fondo a ogni barra e\' il passo intero. '
                 f'La colonna eta porta il termine fisico g, le altre due no: '
                 f'per questo non sono uguali.</text>')

    larghezza_svg = ML + sinistra + larghezza_area + 72 + larghezza_quota + 40
    altezza_svg = y + 48
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "matrix-12-composizione.svg"
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{larghezza_svg}" '
        f'height="{altezza_svg}" viewBox="0 0 {larghezza_svg} {altezza_svg}">\n'
        f'<style>{STILE}</style>\n'
        f'<rect width="{larghezza_svg}" height="{altezza_svg}" '
        f'fill="var(--fondo)"/>\n' + "\n".join(parts) + "\n</svg>\n",
        encoding="utf-8")
    print(f"  {path}")


def fig_thread(rows, outdir):
    """Fase 10: un rank, thread crescenti. Quattro serie, backend x SIMD."""
    data = [r for r in pick(rows, phase="10_matrix_threads")
            if "bind=" not in r["label"] and "omp=0" not in r["label"]]
    if not data:
        return
    grids = valori(data, "nx")
    parts = []
    # Due dimensioni, due canali: la tinta dice il backend, il tratteggio dice
    # se c'e' SIMD. Quattro tinte direbbero la stessa cosa con il doppio dei
    # colori, e due di esse sarebbero vicine.
    for col, grid in enumerate(grids):
        here = pick(data, nx=grid)
        panel = Panel(parts, col, 0, f"Thread, {grid:.0f}^3",
                      "un processo solo, nessun asse diviso",
                      "thread", "ms per passo",
                      [r["threads"] for r in here],
                      [r["wall_ms"] for r in here], xlog=True, ylog=True)
        tacche(panel, valori(here, "threads"))
        voci = []
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
                    etichetta_fine(parts, panel, punti, SERIE[i], backend)
        voci.append((NEUTRO, "tratteggio: senza SIMD", "6,4"))
        base = sorted((r["threads"], r["wall_ms"])
                      for r in pick(here, backend="schur", simd=1.0))
        if base:
            t0, y0 = base[0]
            linea(parts, panel, [(t, y0 * t0 / t) for t, _ in base], NEUTRO,
                  marcatori=False)
            voci.append((NEUTRO, "ideale", ""))
        legenda(parts, panel, voci)
    write(outdir, "matrix-10-thread.svg", len(grids), 1, parts, serie=2)


def fig_forme(rows, outdir):
    """Fase 11: quanto cambia il tempo a seconda di COME si divide.

    Per ogni numero di rank si misurano tutte le terne (px, py, pz). Qui si
    disegna l'intervallo fra la forma migliore e la peggiore: e' quello il
    costo di sbagliare la divisione, ed e' la domanda della fase."""
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
        panel = Panel(parts, col, 0, f"Forma della griglia, {grid:.0f}^3",
                      "intervallo fra la forma migliore e la peggiore",
                      "processi", "ms per passo",
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
                # Lo scostamento laterale separa i due backend: senza, i due
                # intervalli si sovrappongono sullo stesso x e non si leggono.
                x = panel.px(n) + (i - 0.5) * 7
                parts.append(f'<line x1="{x:.1f}" y1="{panel.py(lo):.1f}" '
                             f'x2="{x:.1f}" y2="{panel.py(hi):.1f}" '
                             f'stroke="{colore}" stroke-width="2" '
                             f'stroke-linecap="round"/>')
                parts.append(f'<circle cx="{x:.1f}" cy="{panel.py(lo):.1f}" '
                             f'r="4" fill="{colore}"/>')
                parts.append(f'<circle cx="{x:.1f}" cy="{panel.py(hi):.1f}" '
                             f'r="4" fill="var(--riquadro)" stroke="{colore}" '
                             f'stroke-width="2"/>')
                migliori.append((n, lo))
            if migliori:
                linea(parts, panel, migliori, colore, marcatori=False)
                voci.append((colore, backend, ""))
        legenda(parts, panel, voci)
        parts.append(f'<text x="{panel.x0}" y="{panel.y0 + H + 56}" '
                     f'class="nota">pieno: la forma migliore. vuoto: la '
                     f'peggiore.</text>')
    write(outdir, "matrix-11-forme.svg", len(grids), 1, parts)


def fig_asse_diviso(rows, outdir):
    """Fase 11, la seconda domanda: quale asse conviene dividere.

    Per ogni backend, il tempo medio delle forme che dividono un dato asse.
    La previsione dal codice e' che i due backend diano risposte diverse, e
    questa figura e' dove si vede o non si vede."""
    data = [r for r in pick(rows, phase="11_matrix_mpi", simd=1.0)
            if cubica(r)]
    data = [r for r in data if r["ranks"] and r["ranks"] > 1]
    if not data:
        return
    grids = valori(data, "nx")
    parts = []
    assi = [("px", "x diviso"), ("py", "y diviso"), ("pz", "z diviso")]

    for col, grid in enumerate(grids):
        here = pick(data, nx=grid)
        ranks = valori(here, "ranks")
        panel = Panel(parts, col, 0, f"Quale asse dividere, {grid:.0f}^3",
                      "mediana delle forme che dividono quell'asse",
                      "processi", "ms per passo",
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
                # Le tre serie sono gli assi; il backend e' il tratteggio,
                # cosi' l'identita' non e' affidata al colore due volte.
                linea(parts, panel, punti, SERIE[i],
                      "" if k == 0 else "6,4", marcatori=(k == 0))
                if k == 0:
                    voci.append((SERIE[i], nome, ""))
        voci.append((NEUTRO, "continuo: schur", ""))
        voci.append((NEUTRO, "tratteggio: pipeline", "6,4"))
        legenda(parts, panel, voci)
    write(outdir, "matrix-11-assi.svg", len(grids), 1, parts, serie=3)


def fig_rettangolo(rows, outdir):
    """Fase 12: il rettangolo pieno rank x thread, una mappa per backend."""
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
                       "ms per passo; righe processi, colonne thread",
                       "thread per processo", scala=scala)
        parts.append(f'<text x="{x0 - 52}" y="{y0 + altezza / 2}" class="asse" '
                     f'transform="rotate(-90 {x0 - 52} {y0 + altezza / 2})">'
                     f'processi</text>')
        scala_calore(parts, x0, y0 + altezza + 66, scala[0], scala[1])
    parts.append(f'<text x="{ML}" y="{MT + altezza + 120}" class="nota">'
                 f'Stessa scala di colore nei due riquadri, cosi\' si possono '
                 f'confrontare. Le anti-diagonali a prodotto costante sono le '
                 f'righe della fase 05.</text>')
    width_cols = 2
    write(outdir, "matrix-12-rettangolo.svg", width_cols, 1, parts,
          altezza_extra=70)


def fig_batch(rows, outdir):
    """Fase 13: il batch della pipeline, un riquadro per piazzamento.

    Una serie per riquadro piu' il riferimento Schur in grigio: la domanda e'
    dove sta il minimo di ogni curva, non come si confrontano fra loro, e
    otto curve sovrapposte l'avrebbero nascosta."""
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
        pipe = sorted((g["batch"], g["wall_ms"])
                      for g in pick(qui, backend="pipeline"))
        schur = [g["wall_ms"] for g in pick(qui, backend="schur")]
        if not pipe:
            continue
        ys = [y for _, y in pipe] + schur
        panel = Panel(parts, k % colonne, k // colonne,
                      piazzamento(r, t),
                      f"{grid:.0f}^3, batch della pipeline",
                      "linee per batch", "ms per passo",
                      [x for x, _ in pipe], ys, xlog=True, ylog=False)
        tacche(panel, [x for x, _ in pipe])
        linea(parts, panel, pipe, SERIE[0])
        voci = [(SERIE[0], "pipeline", "")]
        if schur:
            y = min(schur)
            parts.append(f'<line x1="{panel.x0}" y1="{panel.py(y):.1f}" '
                         f'x2="{panel.x0 + W}" y2="{panel.py(y):.1f}" '
                         f'stroke="{NEUTRO}" stroke-width="2"/>')
            voci.append((NEUTRO, "schur, stesso piazzamento", ""))
        # Il minimo, marcato: e' l'unica cosa che il lettore deve portarsi via.
        bx, by = min(pipe, key=lambda p: p[1])
        parts.append(f'<circle cx="{panel.px(bx):.1f}" cy="{panel.py(by):.1f}" '
                     f'r="7" fill="none" stroke="{SERIE[0]}" '
                     f'stroke-width="2"/>')
        parts.append(f'<text x="{panel.px(bx):.1f}" '
                     f'y="{panel.py(by) - 12:.1f}" class="valore" '
                     f'style="fill:{SERIE[0]}">{bx:.0f}</text>')
        legenda(parts, panel, voci)
    parts.append(f'<text x="{ML}" y="{MT + righe * (H + MT + MB) + 10}" '
                 f'class="nota">Ogni riquadro ha la sua scala verticale: la '
                 f'domanda e\' dove sta il minimo di ciascuna curva, non come '
                 f'si confrontano fra loro.</text>')
    write(outdir, "matrix-13-batch.svg", colonne, righe, parts, serie=1,
          altezza_extra=24)


def fig_taglia(rows, outdir):
    """Fase 14: il costo per cella al crescere della taglia.

    Se il lavoro fosse solo aritmetica il costo per cella resterebbe piatto.
    Dove sale, il blocco locale e' uscito dalla cache: e' il muro della banda
    di memoria, e sapere dove sta e' quello che permette di leggere tutte le
    altre figure."""
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
                      "costo per cella e per passo",
                      "lato della griglia", "1e-8 s per cella",
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
        voci.append((NEUTRO, "tratteggio: senza SIMD", "6,4"))
        legenda(parts, panel, voci)
    write(outdir, "matrix-14-taglia.svg", colonne, righe, parts, serie=2)


def fig_memoria(rows, outdir):
    """Fase 14: la memoria di picco, che per la pipeline e' il suo prezzo.

    Tiene c' e d' di tutto il blocco locale per tre componenti: paga in
    memoria quello che Schur paga in aritmetica. Su un asse a se', perche' due
    grandezze non vanno mai su due scale y dello stesso riquadro."""
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
                      "memoria di picco per processo",
                      "lato della griglia", "MB",
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
        legenda(parts, panel, voci, angolo="nw")
    write(outdir, "matrix-14-memoria.svg", len(piazzamenti), 1, parts, serie=2)


def fig_scaling(rows, outdir):
    """Fase 14: scaling forte e debole, con la retta ideale accanto."""
    forte = [r for r in pick(rows, phase="14_matrix_size")
             if " forte " in r["label"]]
    debole = [r for r in pick(rows, phase="14_matrix_size")
              if " debole " in r["label"]]
    if not forte and not debole:
        return
    parts = []

    if forte:
        unita = sorted({max(r["ranks"], r["threads"]) for r in forte})
        panel = Panel(parts, 0, 0, "Scaling forte",
                      "stesso problema, piu' unita' di calcolo",
                      "unita' (processi oppure thread)", "speedup",
                      # Log su entrambi gli assi: cosi' la retta ideale e'
                      # una retta, e lo scarto si legge uguale a 2 unita' e a
                      # 56 invece di schiacciarsi in basso a sinistra.
                      unita, [1, max(unita)], xlog=True, ylog=True)
        tacche(panel, unita)
        voci = []
        i = 0
        for backend in ("schur", "pipeline"):
            for modo, chiave in (("processi", "ranks"), ("thread", "threads")):
                serie = [r for r in forte if r["backend"] == backend
                         and (r["threads"] == 1 if chiave == "ranks"
                              else r["ranks"] == 1)]
                punti = sorted((r[chiave], r["wall_ms"]) for r in serie)
                if len(punti) < 2:
                    continue
                base = punti[0][1]
                linea(parts, panel, [(u, base / w) for u, w in punti],
                      SERIE[i // 2], "" if i % 2 == 0 else "6,4",
                      marcatori=(i % 2 == 0))
                voci.append((SERIE[i // 2], f"{backend}, {modo}",
                             "" if i % 2 == 0 else "6,4"))
                i += 1
        linea(parts, panel, [(u, u) for u in unita], NEUTRO, marcatori=False)
        voci.append((NEUTRO, "ideale", ""))
        legenda(parts, panel, voci, angolo="nw")

    if debole:
        ranks = valori(debole, "ranks")
        panel = Panel(parts, 1, 0, "Scaling debole",
                      "celle per processo costanti",
                      "processi", "efficienza",
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
        voci.append((NEUTRO, "ideale", ""))
        legenda(parts, panel, voci)

    write(outdir, "matrix-14-scaling.svg", 2, 1, parts, serie=2)


def fig_norme(rows, outdir):
    """Fase 15: la campagna risolve ancora il problema giusto?

    Un numero solo, e quindi non un grafico: quanto si discosta la
    configurazione peggiore dal riferimento. Atteso zero, non `piccolo'."""
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
    esito = ("tutte le configurazioni danno la stessa risposta"
             if peggiore == 0 else
             f"una configurazione se ne discosta: {chi}")
    parts = [
        f'<text x="{ML}" y="{MT - 14}" class="titolo">'
        f'Correttezza su tutta la matrice</text>',
        f'<text x="{ML}" y="{MT + 6}" class="sotto">'
        f'{len(data)} configurazioni confrontate con {riferimento["label"]}'
        f'</text>',
        f'<text x="{ML}" y="{MT + 74}" class="enorme" style="fill:{colore}">'
        f'{peggiore:.2e}</text>',
        f'<text x="{ML}" y="{MT + 98}" class="nota">'
        f'scarto relativo massimo sulle norme L2</text>',
        f'<text x="{ML}" y="{MT + 122}" class="nota">{esito}</text>',
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "matrix-15-norme.svg"
    width, height = 640, 200
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        f'<style>{STILE}</style>\n'
        f'<rect width="{width}" height="{height}" fill="var(--fondo)"/>\n'
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
