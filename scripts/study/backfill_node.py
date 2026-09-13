#!/usr/bin/env python3
"""Add the `node` column to the results of a campaign that did not record it.

    ./scripts/study/backfill_node.py                  every phase in build/study
    ./scripts/study/backfill_node.py build/study-old  another base directory

Until the column existed, the node a case ran on was written only in the
phase's run.log: every job prints `node:` (`nodo:` before the logs were
translated) in its machine block, and every case that writes a row prints its
outcome on one line. The k-th outcome line of run.log is the k-th row of
results.csv, so the node can be recovered row by row.

A phase is rewritten only when the match holds for every row -- same count,
same label, same wall time -- and the original stays next to it as
results.csv.before-node. A phase that already has the column is left alone,
so running this twice changes nothing.
"""

import collections
import csv
import os
import re
import shutil
import sys
from pathlib import Path

NODO = re.compile(r'^(?:nodo|node):\s+(\S+)')
# Una misura: `<etichetta>  <n> ms  NxNxN  non contato|unaccounted ...'.
MISURA = re.compile(r'^  (.+?)\s+(-?\d+\.\d+) ms\s+\d+x\d+x\d+\s+'
                    r'(?:non contato|unaccounted)\s')
# Un esito senza misura che scrive comunque una riga, sulla stessa riga.
ESITO = re.compile(r'^  (.+?)\s+(?:timeout|fallito|failed|illeggibile|'
                   r'unreadable) \(\d+s\)\s*$')
# Un fallimento stampa l'uscita del programma prima del suo esito, che finisce
# su una riga a se', lontano dall'etichetta: non si abbina con sicurezza, e la
# fase si rifiuta invece di indovinare.
ORFANO = re.compile(r'^(?:fallito|failed|illeggibile|unreadable) \(\d+s\)')


def esiti(log):
    nodo, trovati, orfani = None, [], 0
    with open(log, errors='replace') as handle:
        for riga in handle:
            riga = riga.rstrip('\n')
            m = NODO.match(riga)
            if m:
                nodo = m.group(1)
                continue
            if ORFANO.match(riga):
                orfani += 1
                continue
            m = MISURA.match(riga)
            if m:
                trovati.append((m.group(1).strip(), m.group(2), nodo))
                continue
            m = ESITO.match(riga)
            if m:
                trovati.append((m.group(1).strip(), None, nodo))
    return trovati, orfani


def main():
    root = Path(__file__).resolve().parents[2]
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'build' / 'study'
    fasi = sorted(d for d in base.iterdir()
                  if (d / 'run.log').is_file() and (d / 'results.csv').is_file())
    if not fasi:
        sys.exit(f'no phase with both run.log and results.csv under {base}')

    rifiutate = 0
    for fase in fasi:
        percorso = fase / 'results.csv'
        with open(percorso, newline='') as handle:
            tabella = list(csv.reader(handle))
        if not tabella:
            print(f'  {fase.name:20s} empty results.csv, skipped')
            continue
        intestazione, righe = tabella[0], tabella[1:]
        if 'node' in intestazione:
            print(f'  {fase.name:20s} already has the node column')
            continue
        if (intestazione[-2:] != ['status', 'note']
                or 'label' not in intestazione or 'wall_ms' not in intestazione):
            print(f'  {fase.name:20s} REFUSED: unexpected header')
            rifiutate += 1
            continue
        i_label = intestazione.index('label')
        i_wall = intestazione.index('wall_ms')

        trovati, orfani = esiti(fase / 'run.log')
        problemi = []
        if orfani:
            problemi.append(f'{orfani} failures reported on a line of their own')
        if len(trovati) != len(righe):
            problemi.append(f'{len(trovati)} outcomes in run.log, '
                            f'{len(righe)} rows in results.csv')
        for k, ((label, wall, nodo), riga) in enumerate(zip(trovati, righe)):
            if (nodo is None or label != riga[i_label]
                    or (wall is not None and wall != riga[i_wall])):
                problemi.append(f'row {k + 1}: log [{label}] {wall} on {nodo}, '
                                f'csv [{riga[i_label]}] {riga[i_wall]}')
                break
        if problemi:
            print(f'  {fase.name:20s} REFUSED: ' + '; '.join(problemi))
            rifiutate += 1
            continue

        copia = fase / 'results.csv.before-node'
        if not copia.exists():
            shutil.copy2(percorso, copia)
        # Il nodo va subito prima di stato e nota, come le colonne aggiunte
        # prima di lui: stato e nota restano il penultimo e l'ultimo campo.
        nuove = [intestazione[:-2] + ['node'] + intestazione[-2:]]
        nuove += [riga[:-2] + [nodo] + riga[-2:]
                  for (_, _, nodo), riga in zip(trovati, righe)]
        provvisorio = fase / 'results.csv.tmp'
        with open(provvisorio, 'w', newline='') as handle:
            csv.writer(handle, lineterminator='\n').writerows(nuove)
        os.replace(provvisorio, percorso)
        conteggio = collections.Counter(nodo for _, _, nodo in trovati)
        print(f'  {fase.name:20s} {len(righe)} rows: '
              + ', '.join(f'{n} {c}' for n, c in sorted(conteggio.items())))
    sys.exit(1 if rifiutate else 0)


if __name__ == '__main__':
    main()
