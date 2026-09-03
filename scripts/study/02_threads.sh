#!/usr/bin/env bash
#PBS -N nsb-02-threads
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=04:00:00
#PBS -j oe
#
# Fase 2 -- i thread da soli: un processo, niente dominio diviso.
#
# Domanda: quanto rende OpenMP quando nessuna linea e' spezzata, e la risposta
# e' la stessa per i due backend?
#
# La previsione, letta nel codice prima di misurare, e' che NON lo sia:
#
#   schur     src/tridiag/schur/momentum.c e pressure.c hanno
#             WORKERS_PARALLEL_FOR sui piani e sulle linee. Con un processo
#             solo whole_axis e' vero ovunque, quindi si spartiscono i piani e
#             i thread lavorano sul pezzo grosso del passo.
#   pipeline  src/tridiag/pipeline/ non contiene nessuna direttiva OpenMP. I
#             thread accelerano soltanto il contorno condiviso (field.c,
#             pressure_common.c), e i tre sistemi di quantita' di moto -- che
#             sono la maggioranza del passo -- restano su un core.
#
# Se e' cosi', la colonna `pipeline' si appiattisce presto e la sua distanza da
# `schur' misura esattamente quanto vale la threadizzazione mancante: e' il
# primo dei due lavori aperti sul backend (README, "Two things the pipeline
# backend does not do yet").
#
# La coda della fase e' il piazzamento: gli stessi 28 thread su un socket solo
# o distribuiti su due sono la stessa potenza di calcolo con meta' o con tutti
# i canali di memoria. Su uno stencil questa differenza e' spesso piu' grande
# di quella fra due algoritmi.
#
#   qsub scripts/study/02_threads.sh
#   GRIDS=128 qsub scripts/study/02_threads.sh     una taglia sola

source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"

cd "${PBS_O_WORKDIR:-$STUDY_ROOT}"
study_begin 02_threads
study_machine

GRIDS="${GRIDS:-128 256}"
THREADS="${THREADS:-1 2 4 7 14 28 56 112}"
# La pipeline non threada i suoi kernel: ogni punto costa quanto il caso
# seriale, e a 256^3 sono minuti. Si tengono i punti che bastano a mostrare
# che la curva e' piatta, non tutta la spazzolata.
THREADS_PIPELINE_BIG="${THREADS_PIPELINE_BIG:-1 7 28 56}"

CASE_RANKS=1
CASE_SHAPE="1 1 1"
CASE_SIMD=1
CASE_OMP=1
CASE_MPI=1
CASE_REPEATS="${REPEATS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-2400}"

for n in $GRIDS; do
    steps="${STEPS:-10}"
    [[ "$n" -le 128 ]] && steps="${STEPS_SMALL:-20}"

    echo "=== ${n}^3, $steps passi: un processo, thread crescenti ==="
    for backend in schur pipeline; do
        threads_list="$THREADS"
        if [[ "$backend" == pipeline && "$n" -ge 224 ]]; then
            threads_list="$THREADS_PIPELINE_BIG"
        fi
        for t in $threads_list; do
            study_case label="$backend ${n}^3 T=$t" backend="$backend" \
                threads="$t" grid="$n $n $n" steps="$steps"
        done
    done
    echo
done

# ------------------------------------------------------------- piazzamento

if [[ "$STUDY_SOCKETS" -gt 1 ]] && command -v numactl > /dev/null; then
    half=$(( STUDY_PHYSICAL / STUDY_SOCKETS ))
    n="${PLACEMENT_GRID:-256}"
    steps="${STEPS:-10}"

    echo "=== $half thread: un socket contro due ==="
    echo "    (stessa potenza di calcolo, meta' o tutti i canali di memoria)"
    for backend in schur pipeline; do
        study_case label="$backend un socket" backend="$backend" \
            threads="$half" grid="$n $n $n" steps="$steps" bind=close \
            wrap="numactl --cpunodebind=0 --membind=0" \
            note="thread e memoria sul solo socket 0"
        study_case label="$backend due socket" backend="$backend" \
            threads="$half" grid="$n $n $n" steps="$steps" bind=spread \
            note="thread distribuiti sui due socket"
    done
    echo
fi

# ------------------------------------------------------------------ risultato

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo "=== risultato ==="
    awk -F, -v phase=02_threads '
    NR == 1 || $1 != phase || $32 != "ok" { next }
    $2 ~ /socket/ { place[$3 "," $2] = $17; next }
    {
        grid = $10; backend = $3; t = $9
        wall[backend "," grid "," t] = $17
        untimed[backend "," grid "," t] = $27
        mom[backend "," grid "," t] = $19 + $20 + $21
        if (!(grid in seen_grid)) { grids[++g] = grid; seen_grid[grid] = 1 }
        if (!(t in seen_t)) { threads[++nt] = t; seen_t[t] = 1 }
    }
    END {
        for (i = 1; i <= g; i++) {
            grid = grids[i]
            printf "\n  %s^3\n", grid
            printf "  %-8s %28s %28s\n", "", "schur", "pipeline"
            printf "  %-8s %11s %8s %8s %11s %8s %8s\n",
                   "thread", "ms/passo", "speedup", "moment.",
                   "ms/passo", "speedup", "moment."
            for (j = 1; j <= nt; j++) {
                t = threads[j]
                a = wall["schur," grid "," t]; b = wall["pipeline," grid "," t]
                if (a == "" && b == "") continue
                if (base_a[grid] == "" && a != "") base_a[grid] = a
                if (base_b[grid] == "" && b != "") base_b[grid] = b
                printf "  %-8s", t
                if (a != "") printf " %11.1f %7.2fx %8.1f", a, base_a[grid]/a, mom["schur," grid "," t]
                else         printf " %11s %8s %8s", "-", "-", "-"
                if (b != "") printf " %11.1f %7.2fx %8.1f", b, base_b[grid]/b, mom["pipeline," grid "," t]
                else         printf " %11s %8s %8s", "-", "-", "-"
                printf "\n"
            }
        }
        if (length(place) > 0) {
            printf "\n  piazzamento\n"
            for (k in place) {
                split(k, p, ",")
                printf "  %-9s %-16s %10.1f ms\n", p[1], p[2], place[k]
            }
        }
        print ""
        print "  La colonna `moment.\x27 e\x27 la somma di eta+zeta+u, cioe\x27 i tre"
        print "  sistemi di quantita\x27 di moto: e\x27 li\x27 che i due backend fanno"
        print "  cose diverse, e li\x27 che si vede se i thread arrivano o no."
    }' "$STUDY_CSV"
fi

study_end
