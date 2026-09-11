#!/usr/bin/env python3
"""I grafici del confronto fra due revisioni, dal CSV di run_patch_ab.sh.

    ./scripts/run_patch_ab.sh HEAD~1 HEAD
    ./scripts/plot_patch_ab.py            legge build/patch-ab/results.csv

Due figure, e rispondono a due domande diverse:

    patch-ab-stadi.svg     dove stava il tempo prima e dove sta adesso, stadio
                           per stadio. Le barrette vanno da prima a dopo.
    patch-ab-guadagno.svg  quanto si e' risparmiato in percentuale, e -- nella
                           stessa figura -- quanto si muovono gli stadi che la
                           modifica non tocca. Quelli sono il rumore, e senza
                           averli accanto una percentuale non si sa leggere.

Il colore qui non dice l'identita' ma il verso: blu dove il tempo e' calato,
rosso dove e' cresciuto, grigio in mezzo. E' una scala divergente, non
categorica, perche' la domanda e' il segno.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_matrix import STILE, NEUTRO, ML, MT  # noqa: E402

MEGLIO = "#2a78d6"
PEGGIO = "#e34948"

# Gli stadi, nell'ordine in cui il passo li esegue. La coppia e' (colonna,
# nome leggibile); `non contato' non e' uno stadio ma quello che avanza, e
# resta in fondo perche' e' li' che si guarda quando i conti non tornano.
STADI = [
    ("eta_ms", "eta, tutto"),
    ("g_ms", "  g, termine noto"),
    ("eta_solve", "  eta, il sistema"),
    ("zeta_ms", "zeta, asse y"),
    ("u_ms", "u, asse z"),
    ("psi_ms", "psi"),
    ("philow_ms", "phi basso"),
    ("phihigh_ms", "phi alto"),
    ("pressure_ms", "pressione"),
    ("porosity_ms", "porosita'"),
    ("untimed_ms", "non contato"),
    ("mpi_ms", "dentro MPI"),
]

# Gli stadi che la modifica di questa serie NON tocca: sono il controllo.
CONTROLLO = {"zeta_ms", "u_ms", "psi_ms", "philow_ms", "phihigh_ms",
             "pressure_ms"}

# Righe che sono gia' dentro un'altra: nel totale non vanno contate due volte.
DENTRO_ETA = {"g_ms", "eta_solve"}

RIGA = 22          # altezza di una riga di stadio
LARGA = 300        # larghezza dell'area di disegno
SINISTRA = 108     # spazio per i nomi degli stadi


def load(path):
    if not path.exists():
        sys.exit(f"manca {path}\n  ./scripts/run_patch_ab.sh")
    righe = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            riga = dict(raw)
            for key, _ in STADI:
                try:
                    riga[key] = float(raw.get(key) or 0.0)
                except ValueError:
                    riga[key] = 0.0
            try:
                riga["wall_ms"] = float(raw.get("wall_ms") or 0.0)
            except ValueError:
                continue
            riga["eta_solve"] = max(riga.get("eta_ms", 0.0)
                                    - riga.get("g_ms", 0.0), 0.0)
            if riga["wall_ms"] <= 0:
                continue
            righe.append(riga)
    print(f"{len(righe)} corse da {path}")
    return righe


def mediana(valori):
    valori = sorted(valori)
    if not valori:
        return None
    meta = len(valori) // 2
    if len(valori) % 2:
        return valori[meta]
    return (valori[meta - 1] + valori[meta]) / 2


def casi(righe):
    """Le configurazioni misurate, ognuna con la mediana di ogni stadio nelle
    due revisioni. La mediana e non la migliore: qui interessa il centro delle
    corse, non la piu' fortunata."""
    fuori = []
    visti = []
    for riga in righe:
        chiave = (riga["backend"], riga["simd"],
                  f'{riga["ranks"]}x{riga["threads"]}', riga["nx"])
        if chiave not in visti:
            visti.append(chiave)
    for chiave in visti:
        caso = {"chiave": chiave, "prima": {}, "dopo": {}}
        for quando in ("prima", "dopo"):
            gruppo = [r for r in righe
                      if (r["backend"], r["simd"],
                          f'{r["ranks"]}x{r["threads"]}', r["nx"]) == chiave
                      and r["revisione"] == quando]
            for key, _ in STADI:
                caso[quando][key] = mediana([r[key] for r in gruppo])
            caso[quando]["wall_ms"] = mediana([r["wall_ms"] for r in gruppo])
        if caso["prima"]["wall_ms"] and caso["dopo"]["wall_ms"]:
            fuori.append(caso)
    return fuori


def titolo_caso(chiave):
    backend, simd, piazzamento, n = chiave
    return f"{backend}, {piazzamento}, {n}^3, simd={simd}"


def intestazione(parts, x, y, titolo, sottotitolo):
    parts.append(f'<text x="{x}" y="{y - 26}" class="titolo">{titolo}</text>')
    parts.append(f'<text x="{x}" y="{y - 10}" class="sotto">{sottotitolo}'
                 f'</text>')


def scrivi(outdir, nome, larghezza, altezza, parts):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / nome
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{larghezza}" '
        f'height="{altezza}" viewBox="0 0 {larghezza} {altezza}">\n'
        f'<style>{STILE}</style>\n'
        f'<rect width="{larghezza}" height="{altezza}" fill="var(--fondo)"/>\n'
        + "\n".join(parts) + "\n</svg>\n", encoding="utf-8")
    print(f"  {path}")


def fig_stadi(lista, outdir, sha_prima, sha_dopo):
    """Da prima a dopo, stadio per stadio, in millisecondi.

    Una barretta per stadio: pallino vuoto dove stava, pieno dove sta. La
    lunghezza della barretta e' il guadagno in millisecondi, che e' la cosa che
    conta quando si decide se una modifica vale la pena."""
    if not lista:
        return
    colonne = min(3, len(lista))
    righe_fig = (len(lista) + colonne - 1) // colonne
    passo_x = SINISTRA + LARGA + 70
    passo_y = len(STADI) * RIGA + 96
    parts = []

    for k, caso in enumerate(lista):
        x0 = ML + (k % colonne) * passo_x + SINISTRA
        y0 = MT + (k // colonne) * passo_y
        massimo = max([v for quando in ("prima", "dopo")
                       for v in caso[quando].values() if v] + [1e-9])

        intestazione(parts, x0 - SINISTRA, y0, titolo_caso(caso["chiave"]),
                     "millisecondi per passo, da prima a dopo")
        for i, (key, nome) in enumerate(STADI):
            y = y0 + i * RIGA + RIGA / 2
            a = caso["prima"].get(key) or 0.0
            b = caso["dopo"].get(key) or 0.0
            xa = x0 + LARGA * a / massimo
            xb = x0 + LARGA * b / massimo
            colore = MEGLIO if b < a else (PEGGIO if b > a else NEUTRO)
            stile = ' font-style="italic"' if key in CONTROLLO else ""
            parts.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="tacca-y"'
                         f'{stile}>{nome}</text>')
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + LARGA}" '
                         f'y2="{y:.1f}" class="griglia"/>')
            parts.append(f'<line x1="{xa:.1f}" y1="{y:.1f}" x2="{xb:.1f}" '
                         f'y2="{y:.1f}" stroke="{colore}" stroke-width="3" '
                         f'stroke-linecap="round"/>')
            parts.append(f'<circle cx="{xa:.1f}" cy="{y:.1f}" r="4.5" '
                         f'fill="var(--fondo)" stroke="{colore}" '
                         f'stroke-width="2"/>')
            parts.append(f'<circle cx="{xb:.1f}" cy="{y:.1f}" r="4.5" '
                         f'fill="{colore}"/>')
            parts.append(f'<text x="{x0 + LARGA + 8}" y="{y + 4:.1f}" '
                         f'class="valore" text-anchor="start">{b:.2f}</text>')
        y = y0 + len(STADI) * RIGA + 16
        parts.append(f'<text x="{x0}" y="{y}" class="tacca-x" '
                     f'text-anchor="start">0</text>')
        parts.append(f'<text x="{x0 + LARGA}" y="{y}" class="tacca-x" '
                     f'text-anchor="end">{massimo:.1f} ms</text>')

    altezza = MT + righe_fig * passo_y + 40
    larghezza = ML + colonne * passo_x
    parts.append(f'<text x="{ML}" y="{altezza - 18}" class="nota">'
                 f'Vuoto: {sha_prima}. Pieno: {sha_dopo}. In corsivo gli stadi '
                 f'che la modifica non tocca.</text>')
    scrivi(outdir, "patch-ab-stadi.svg", larghezza, altezza, parts)


def fig_guadagno(lista, outdir, sha_prima, sha_dopo):
    """La percentuale risparmiata, con il controllo nella stessa figura.

    Barre a partire dallo zero: a destra dove il tempo e' calato, a sinistra
    dove e' cresciuto. Gli stadi che la modifica non tocca sono in grigio, e
    la loro escursione e' il rumore: un guadagno piu' corto di quelle barre
    grigie non e' un guadagno."""
    if not lista:
        return
    colonne = min(3, len(lista))
    righe_fig = (len(lista) + colonne - 1) // colonne
    meta = LARGA / 2
    passo_x = SINISTRA + LARGA + 40
    passo_y = len(STADI) * RIGA + 96
    parts = []

    for k, caso in enumerate(lista):
        x0 = ML + (k % colonne) * passo_x + SINISTRA
        y0 = MT + (k // colonne) * passo_y
        variazioni = []
        for key, nome in STADI:
            a = caso["prima"].get(key) or 0.0
            b = caso["dopo"].get(key) or 0.0
            variazioni.append(100 * (a - b) / a if a > 0 else 0.0)
        limite = max(10.0, max(abs(v) for v in variazioni) * 1.1)

        intestazione(parts, x0 - SINISTRA, y0, titolo_caso(caso["chiave"]),
                     "percento risparmiato; a sinistra dello zero e' peggiorato")
        parts.append(f'<line x1="{x0 + meta}" y1="{y0 - 4}" '
                     f'x2="{x0 + meta}" y2="{y0 + len(STADI) * RIGA}" '
                     f'stroke="var(--linea)" stroke-width="1"/>')
        for i, ((key, nome), v) in enumerate(zip(STADI, variazioni)):
            y = y0 + i * RIGA + RIGA / 2
            larghezza_barra = meta * v / limite
            colore = (NEUTRO if key in CONTROLLO
                      else (MEGLIO if v >= 0 else PEGGIO))
            x = x0 + meta + (0 if v >= 0 else larghezza_barra)
            parts.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" '
                         f'class="tacca-y">{nome}</text>')
            parts.append(f'<rect x="{x:.1f}" y="{y - 7:.1f}" '
                         f'width="{abs(larghezza_barra):.1f}" height="14" '
                         f'rx="3" fill="{colore}"/>')
            # L'etichetta va dentro la barra quando c'e' spazio, fuori quando
            # non ce n'e': una barra lunga spinge il numero fin sopra i nomi
            # degli stadi, e li' non si legge piu' niente.
            fine = x0 + meta + larghezza_barra
            dentro = abs(larghezza_barra) > 42
            if dentro:
                ancora = "end" if v >= 0 else "start"
                scarto = -7 if v >= 0 else 7
                colore_testo = "#ffffff"
            else:
                ancora = "start" if v >= 0 else "end"
                scarto = 8 if v >= 0 else -8
                colore_testo = "var(--inchiostro-2)"
            parts.append(f'<text x="{fine + scarto:.1f}" y="{y + 4:.1f}" '
                         f'class="valore" text-anchor="{ancora}" '
                         f'style="fill:{colore_testo}">{v:+.0f}%</text>')
        y = y0 + len(STADI) * RIGA + 16
        parts.append(f'<text x="{x0 + meta}" y="{y}" class="tacca-x">0</text>')

    altezza = MT + righe_fig * passo_y + 40
    larghezza = ML + colonne * passo_x
    parts.append(f'<text x="{ML}" y="{altezza - 18}" class="nota">'
                 f'{sha_prima} contro {sha_dopo}. In grigio gli stadi che la '
                 f'modifica non tocca: la loro escursione e\' il rumore del '
                 f'nodo, e un guadagno piu\' corto di quelle barre non e\' un '
                 f'guadagno.</text>')
    scrivi(outdir, "patch-ab-guadagno.svg", larghezza, altezza, parts)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", type=Path,
                        default=root / "build" / "patch-ab" / "results.csv")
    parser.add_argument("-o", "--outdir", type=Path,
                        default=root / "docs" / "scaling" / "patch-ab")
    args = parser.parse_args()

    righe = load(args.csv)
    if not righe:
        return
    sha_prima = next((r["sha"] for r in righe if r["revisione"] == "prima"), "?")
    sha_dopo = next((r["sha"] for r in righe if r["revisione"] == "dopo"), "?")
    lista = casi(righe)
    fig_stadi(lista, args.outdir, sha_prima, sha_dopo)
    fig_guadagno(lista, args.outdir, sha_prima, sha_dopo)


if __name__ == "__main__":
    main()
