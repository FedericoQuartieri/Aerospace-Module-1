#!/usr/bin/env bash
#PBS -N nsb-01-ceiling
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Fase 1 -- i tetti, e il costo di ogni pezzo della build a un core solo.
#
# Domanda: qual e' il massimo che questa macchina puo' dare, e quanto costa
# ciascuno dei quattro interruttori (SIMD, OpenMP, MPI, backend) quando non
# c'e' nessun parallelismo da sfruttare?
#
# Perche' serve prima delle altre fasi. Uno speedup letto contro il numero di
# core e' quasi sempre una delusione; letto contro il massimo misurato e'
# un'informazione. Sul portatile il tetto era 2.91x su 4 core, e il 2.18x
# ottenuto era il 75% del possibile, non un 55% mancato (MULTITHREAD.md §7.1).
#
# I tetti sono due perche' il solutore tocca due limiti diversi:
#
#   trig    solo calcolo. E' il tetto della forzante e della permeabilita',
#           che sono trigonometria pura.
#   triad   solo banda di memoria. E' il tetto vero di uno stencil, e i thread
#           lo saturano molto prima di finire i core.
#
# La seconda meta' della fase misura il costo a un thread solo di ogni
# interruttore: la build OpenMP costa circa l'1% anche quando i thread sono
# uno (MULTITHREAD.md §7.3), e senza questa riga quel punto percentuale finisce
# per sbaglio nel conto dello speedup.
#
#   qsub scripts/study/01_ceiling.sh

# PBS non esegue questo file dove sta: ne mette una COPIA nella propria
# directory di spool e lancia quella. $BASH_SOURCE punta li', quindi non dice
# niente su dove sia il repo -- a dirlo e' PBS_O_WORKDIR, la directory da cui
# si e' fatto qsub. Cercare lib.sh accanto allo script funziona da riga di
# comando e fallisce dentro un job, prima ancora che esista un log in cui
# vederlo.
cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub va fatto dalla radice del repo; qui sono in $PWD" >&2
    exit 1
fi

source scripts/study/lib.sh
study_begin 01_ceiling
study_machine

THREADS="${THREADS:-1 2 4 7 14 28 56 112}"
SMALL_GRID="${SMALL_GRID:-128}"
BIG_GRID="${BIG_GRID:-256}"
TRIG_MILLIONS="${TRIG_MILLIONS:-400}"
TRIAD_MB="${TRIAD_MB:-2048}"
CEILINGS="$STUDY_OUT/ceilings.csv"

# --------------------------------------------------------------------- tetti

micro="$STUDY_BIN/ceiling"
if [[ ! -x "$micro" || "${REBUILD:-0}" == "1" ]]; then
    printf 'compilo il microbenchmark  '
    cc -std=gnu11 -O3 -fopenmp -o "$micro" \
        "$STUDY_ROOT/scripts/study/micro/ceiling.c" -lm
    echo ok
fi

[[ -f "$CEILINGS" ]] || printf 'kind,threads,time_ms,rate,unit\n' > "$CEILINGS"

ceiling_run()
{
    local kind="$1" threads="$2" size="$3"

    if grep -q "^$kind,$threads," "$CEILINGS" && [[ "${RESUME:-1}" == "1" ]]; then
        printf '  %-6s %3s thread   gia\x27 fatto\n' "$kind" "$threads"
        return 0
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '  %-6s %3s thread   (dry run)\n' "$kind" "$threads"
        return 0
    fi

    local out
    # spread, non close: il tetto della banda si vede solo distribuendo i
    # thread su tutti i socket, cioe' su tutti i canali di memoria.
    out="$(OMP_NUM_THREADS="$threads" OMP_PLACES=cores OMP_PROC_BIND=spread \
           "$micro" "$kind" "$size")"
    local time_ms rate unit
    time_ms="$(sed 's/.*time_ms=\([^ ]*\).*/\1/' <<< "$out")"
    rate="$(sed 's/.*rate=\([^ ]*\).*/\1/' <<< "$out")"
    unit="$(sed 's/.*unit=\([^ ]*\).*/\1/' <<< "$out")"
    printf '%s,%s,%s,%s,%s\n' "$kind" "$threads" "$time_ms" "$rate" "$unit" \
        >> "$CEILINGS"
    printf '  %-6s %3s thread %12s ms %12s %s\n' \
        "$kind" "$threads" "$time_ms" "$rate" "$unit"
}

echo "=== tetto di calcolo: trigonometria, nessun accesso a memoria ==="
for t in $THREADS; do ceiling_run trig "$t" "$TRIG_MILLIONS"; done
echo
echo "=== tetto di banda: triad su ${TRIAD_MB} MB per array ==="
for t in $THREADS; do ceiling_run triad "$t" "$TRIAD_MB"; done
echo

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    awk -F, '
    NR == 1 { next }
    { rate[$1 "," $2] = $4; unit[$1] = $5; if (!($2 in seen)) { order[++n] = $2; seen[$2] = 1 } }
    END {
        printf "  %-8s %14s %9s %14s %9s\n",
               "thread", "trig", "speedup", "triad", "speedup"
        for (i = 1; i <= n; i++) {
            t = order[i]
            a = rate["trig," t] + 0; b = rate["triad," t] + 0
            if (i == 1) { base_a = a; base_b = b }
            printf "  %-8s %10.1f %s %7.2fx %10.1f %s %7.2fx\n",
                   t, a, unit["trig"], (base_a ? a / base_a : 0),
                      b, unit["triad"], (base_b ? b / base_b : 0)
        }
        print ""
        print "  Il primo numero e\x27 il tetto dei pezzi che calcolano, il secondo"
        print "  quello dei pezzi che spostano memoria. Nessuno stadio del solutore"
        print "  puo\x27 scalare piu\x27 del minore dei due che lo riguardano."
    }' "$CEILINGS"
fi
echo

# ------------------------------------------------- costo dei quattro flag a 1

CASE_REPEATS="${REPEATS:-3}"
CASE_TIMEOUT="${CASE_TIMEOUT:-900}"
CASE_RANKS=1
CASE_THREADS=1

echo "=== un core solo: quanto costa ciascun interruttore ==="
echo "    (${SMALL_GRID}^3, ${STEPS_SMALL:-10} passi, tre ripetizioni, la migliore)"

for backend in schur pipeline; do
    for simd in 0 1; do
        for omp in 0 1; do
            for mpi in 0 1; do
                study_case label="$backend simd=$simd omp=$omp mpi=$mpi" \
                    backend="$backend" simd="$simd" omp="$omp" mpi="$mpi" \
                    grid="$SMALL_GRID $SMALL_GRID $SMALL_GRID" \
                    steps="${STEPS_SMALL:-10}" \
                    shape="1 1 1"
            done
        done
    done
done

echo
echo "=== un core solo, ${BIG_GRID}^3: il denominatore degli speedup delle altre fasi ==="
for backend in schur pipeline; do
    study_case label="$backend $BIG_GRID baseline" backend="$backend" \
        simd=1 omp=1 mpi=1 grid="$BIG_GRID $BIG_GRID $BIG_GRID" \
        steps="${STEPS_BIG:-10}" shape="1 1 1" repeats=2
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== risultato ==="
    awk -F, -v phase=01_ceiling -v small="$SMALL_GRID" '
    NR == 1 || $1 != phase || $32 != "ok" || $10 != small { next }
    {
        key = $3 "," $5 "," $6 "," $7
        wall[key] = $17; eta[key] = $19; zeta[key] = $20; u[key] = $21
        if (!($3 in backends)) { backends[$3] = 1; order[++n] = $3 }
    }
    END {
        printf "\n  %-9s %-6s %-5s %-5s %10s %9s %9s %9s\n",
               "backend", "simd", "omp", "mpi", "ms/passo", "eta", "zeta", "u"
        for (i = 1; i <= n; i++) {
            b = order[i]
            for (simd = 0; simd <= 1; simd++)
            for (omp = 0; omp <= 1; omp++)
            for (mpi = 0; mpi <= 1; mpi++) {
                k = b "," simd "," omp "," mpi
                if (k in wall)
                    printf "  %-9s %-6s %-5s %-5s %10.1f %9.1f %9.1f %9.1f\n",
                           b, simd, omp, mpi, wall[k], eta[k], zeta[k], u[k]
            }
        }
        print ""
        print "  Da leggere per differenze, non per valore assoluto:"
        print "    simd 0 -> 1   quanto rendono i kernel vettorizzati (solo schur,"
        print "                  e solo su zeta e u: la pipeline non ne ha)"
        print "    omp  0 -> 1   quanto costa la build a thread, con UN thread"
        print "    mpi  0 -> 1   quanto costa la build MPI, con UN processo"
        print "    schur vs pipeline   il costo di algoritmo a processo singolo,"
        print "                  dove nessuna linea e\x27 divisa e nessuno comunica"
    }' "$STUDY_CSV"
fi

study_end
