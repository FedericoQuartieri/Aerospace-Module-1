#!/usr/bin/env bash
#PBS -N nsb-04-shape
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Fase 4 -- quale asse viene diviso, a parita' di tutto il resto.
#
# Domanda: 8 processi disposti 8x1x1, 1x1x8 o 2x2x2 fanno lo stesso lavoro su
# blocchi della stessa dimensione. Perche' non costano uguale?
#
# Perche' i tre assi non sono intercambiabili, in nessuno dei due backend:
#
#   il passo X (eta) e' il piu' costoso di tutti -- da solo il 77% del passo
#   temporale (MULTITHREAD.md §10) -- e non ha kernel SIMD nemmeno quando
#   l'asse resta intero. Dividerlo tocca il pezzo grosso.
#
#   i passi Y e Z (zeta, u) hanno i kernel vettorizzati, ma valgono solo su
#   una linea intera: appena il loro asse e' diviso, spariscono. Dividere Y o
#   Z costa quindi una cosa che dividere X non costa.
#
#   per la pipeline l'asse diviso decide quante linee indipendenti ci sono da
#   mandare in catena e quanti stadi ha la catena: e' la variabile della
#   formula di PIPELINE.md §1.5, non un dettaglio di disposizione.
#
# Nessuna misura precedente ha toccato questa variabile: MPI_Dims_create
# sceglieva, e nessuno la contraddiceva. E' l'unica fase che richiede il
# binario nuovo (test/bench.c), che accetta la forma sulla riga di comando.
#
# La griglia e' 224^3 perche' 224 = 7 x 32: ogni forma di questa fase la divide
# esattamente, e nessuna riga porta con se' dello sbilanciamento.
#
#   qsub scripts/study/04_shape.sh

source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"

cd "${PBS_O_WORKDIR:-$STUDY_ROOT}"
study_begin 04_shape
study_machine

GRID="${GRID:-224 224 224}"
STEPS="${STEPS:-10}"

CASE_THREADS=1
CASE_OMP=0
CASE_MPI=1
CASE_SIMD=1
CASE_GRID="$GRID"
CASE_STEPS="$STEPS"
CASE_REPEATS="${REPEATS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-900}"

# Per ogni conteggio di rank: i tre casi degeneri (un asse solo diviso, uno per
# asse) piu' le disposizioni intermedie. I degeneri isolano l'effetto di un
# asse alla volta, gli altri dicono se dividerne due o tre insieme costa piu'
# o meno della somma.
declare -A SHAPES
SHAPES[2]="2-1-1 1-2-1 1-1-2"
SHAPES[4]="4-1-1 1-4-1 1-1-4 2-2-1 2-1-2 1-2-2"
SHAPES[8]="8-1-1 1-8-1 1-1-8 2-2-2 4-2-1 2-1-4"
SHAPES[56]="56-1-1 1-56-1 1-1-56 7-4-2 2-2-14 14-2-2 4-14-1"

for ranks in ${RANKS:-4 8 56}; do
    [[ -z "${SHAPES[$ranks]:-}" ]] && continue
    echo "=== $ranks processi, ${GRID// /x}, $STEPS passi ==="
    for backend in schur pipeline; do
        for shape in ${SHAPES[$ranks]}; do
            study_case label="$backend ${ranks}p ${shape}" \
                backend="$backend" ranks="$ranks" shape="${shape//-/ }"
        done
    done
    echo
done

# Senza SIMD la differenza fra gli assi resta solo quella dell'algoritmo: il
# confronto con le righe di sopra separa "vettorizzazione persa" da "costo di
# dividere".
nosimd_ranks="${NOSIMD_RANKS:-8}"
if [[ -n "${SHAPES[$nosimd_ranks]:-}" ]]; then
    echo "=== $nosimd_ranks processi senza SIMD: cosa resta della differenza fra gli assi ==="
    for backend in schur pipeline; do
        for shape in ${SHAPES[$nosimd_ranks]}; do
            study_case label="$backend ${nosimd_ranks}p ${shape} simd=0" simd=0 \
                backend="$backend" ranks="$nosimd_ranks" shape="${shape//-/ }"
        done
    done
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== risultato ==="
    awk -F, -v phase=04_shape '
    NR == 1 || $1 != phase || $32 != "ok" { next }
    {
        form = $14 "x" $15 "x" $16
        k = $3 "," $8 "," $5 "," form
        wall[k] = $17; mpims[k] = $18
        eta[k] = $19; zeta[k] = $20; u[k] = $21
        gk = $8 "," $5
        if (!(gk in seen_g)) { gs[++ng] = gk; seen_g[gk] = 1 }
        if (!(gk "," form in seen_f)) {
            forms[gk] = forms[gk] " " form; seen_f[gk "," form] = 1
        }
    }
    END {
        for (i = 1; i <= ng; i++) {
            gk = gs[i]; split(gk, p, ",")
            printf "\n  %s processi, simd=%s\n", p[1], p[2]
            printf "  %-10s %31s %31s\n", "", "schur", "pipeline"
            printf "  %-10s %10s %6s %6s %6s %10s %6s %6s %6s\n",
                   "forma", "ms/passo", "eta", "zeta", "u",
                   "ms/passo", "eta", "zeta", "u"
            n = split(forms[gk], list, " ")
            best_a = 0; best_b = 0
            for (j = 1; j <= n; j++) {
                f = list[j]
                ka = "schur," gk "," f; kb = "pipeline," gk "," f
                printf "  %-10s", f
                if (ka in wall) {
                    printf " %10.1f %6.0f %6.0f %6.0f", wall[ka], eta[ka], zeta[ka], u[ka]
                    if (best_a == 0 || wall[ka] < best_a) { best_a = wall[ka]; bf_a = f }
                    if (worst_a < wall[ka]) { worst_a = wall[ka]; wf_a = f }
                } else printf " %10s %6s %6s %6s", "-", "-", "-", "-"
                if (kb in wall) {
                    printf " %10.1f %6.0f %6.0f %6.0f", wall[kb], eta[kb], zeta[kb], u[kb]
                    if (best_b == 0 || wall[kb] < best_b) { best_b = wall[kb]; bf_b = f }
                    if (worst_b < wall[kb]) { worst_b = wall[kb]; wf_b = f }
                } else printf " %10s %6s %6s %6s", "-", "-", "-", "-"
                printf "\n"
            }
            if (best_a > 0)
                printf "    schur:    migliore %s, peggiore %s -> %.2fx di differenza\n",
                       bf_a, wf_a, worst_a / best_a
            if (best_b > 0)
                printf "    pipeline: migliore %s, peggiore %s -> %.2fx di differenza\n",
                       bf_b, wf_b, worst_b / best_b
            worst_a = 0; worst_b = 0
        }
        print ""
        print "  Le colonne eta/zeta/u sono i tre sistemi direzionali. La forma"
        print "  che divide l\x27asse X gonfia eta; quelle che dividono Y o Z"
        print "  gonfiano zeta o u, e su schur ci aggiungono la SIMD perduta."
    }' "$STUDY_CSV"
fi

study_end
