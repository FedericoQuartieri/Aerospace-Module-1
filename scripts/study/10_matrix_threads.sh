#!/usr/bin/env bash
#PBS -N nsb-10-threads
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Fase 10 -- un rank solo, thread crescenti, per tutte le varianti.
#
# La 02 chiede se i thread rendono. Questa riempie la riga: i due backend, con
# e senza SIMD, su tre taglie, da un thread a tutte le cpu logiche. Serve a
# separare tre cose che nelle misure vecchie stavano insieme:
#
#   - quanto scala ogni backend da solo, senza dominio diviso;
#   - quanto di quello scaling e' SIMD e quanto sono thread, che non si
#     sommano: i kernel vettoriali saturano la banda prima dei thread;
#   - dove si ferma, e se il punto in cui si ferma dipende dalla taglia. Su
#     uno stencil il tetto e' la banda di memoria, e la banda per core
#     peggiora con la taglia.
#
# Le righe con OMP=0 sono il riferimento: la build a thread costa circa l'1%
# anche quando i thread sono uno, e senza quelle righe quell'1% finisce dentro
# lo speedup.
#
# La coda del piazzamento: gli stessi 28 thread su un socket o distribuiti su
# due sono la stessa potenza di calcolo con meta' o con tutti i canali di
# memoria. Su uno stencil questa differenza e' spesso piu' grande di quella
# fra due algoritmi.
#
#   qsub scripts/study/10_matrix_threads.sh
#   GRIDS=128 qsub scripts/study/10_matrix_threads.sh      una taglia sola

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub va fatto dalla radice del repo; qui sono in $PWD" >&2
    exit 1
fi

source scripts/study/lib.sh
source scripts/study/matrix.sh

STUDY_CHAIN_MAX="$MATRIX_CHAIN_MAX"
study_begin 10_matrix_threads
study_machine

GRIDS="${GRIDS:-64 128 224}"
THREADS="${THREADS:-$MATRIX_THREADS}"

CASE_RANKS=1
CASE_MPI=1
CASE_OMP=1
CASE_SHAPE="1 1 1"
CASE_REPEATS="${REPEATS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-1200}"

for n in $GRIDS; do
    steps="$(matrix_steps "$n")"

    echo "=== ${n}^3, $steps passi: un rank, thread crescenti ==="
    for backend in $MATRIX_BACKENDS; do
        for simd in $MATRIX_SIMD; do
            # Il riferimento senza OpenMP: stesso codice, nessun thread
            # compilato dentro.
            study_case label="$backend ${n}^3 s$simd omp=0" \
                backend="$backend" simd="$simd" omp=0 threads=1 \
                grid="$n $n $n" steps="$steps"

            for t in $THREADS; do
                [[ "$t" -le "$(matrix_units)" ]] || continue
                study_case label="$backend ${n}^3 s$simd T=$t" \
                    backend="$backend" simd="$simd" threads="$t" \
                    grid="$n $n $n" steps="$steps"
            done
        done
    done
    echo
done

# Il piazzamento, a parita' di thread: `close' li stringe sul primo socket
# finche' ci stanno, `spread' li distribuisce su tutti. Meta' dei canali di
# memoria contro tutti.
BIND_GRID="${BIND_GRID:-224}"
BIND_THREADS="${BIND_THREADS:-14 28}"
echo "=== ${BIND_GRID}^3: gli stessi thread, stretti o distribuiti ==="
for backend in $MATRIX_BACKENDS; do
    for t in $BIND_THREADS; do
        for bind in close spread; do
            study_case label="$backend bind=$bind T=$t" \
                backend="$backend" simd=1 threads="$t" bind="$bind" \
                grid="$BIND_GRID $BIND_GRID $BIND_GRID" \
                steps="$(matrix_steps "$BIND_GRID")" \
                note="piazzamento $bind"
        done
    done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== risultato: ms/passo e speedup sul proprio caso a 1 thread ==="
    # La lista dei thread arriva dalla shell: ordinarla dentro awk vorrebbe
    # asort, che e' di gawk, e sul cluster awk puo' essere mawk.
    awk -F, -v phase=10_matrix_threads -v tlist="$THREADS" '
    NR == 1 || $1 != phase || $(NF - 1) != "ok" || $2 ~ /bind=/ { next }
    {
        key = $3 "," $5 "," $10
        if ($2 ~ /omp=0/) { base[key] = $17; next }
        wall[key "," $9] = $17
        if (!(key in seen)) { keys[++nk] = key; seen[key] = 1 }
    }
    END {
        n = split(tlist, ts, " ")
        printf "  %-10s %-5s %-6s %10s", "backend", "simd", "griglia", "omp=0"
        for (i = 1; i <= n; i++) printf " %8s", "T=" ts[i]
        printf "\n"
        for (k = 1; k <= nk; k++) {
            split(keys[k], p, ",")
            printf "  %-10s %-5s %-6s %10s", p[1], p[2], p[3],
                   (keys[k] in base ? sprintf("%.0f", base[keys[k]]) : "-")
            for (i = 1; i <= n; i++) {
                kk = keys[k] "," ts[i]
                if (kk in wall) printf " %8.1f", wall[kk]
                else printf " %8s", "-"
            }
            printf "\n"
        }
        print ""
        print "  I numeri sono ms per passo temporale, il migliore delle ripetizioni."
        print "  La colonna omp=0 e\x27 lo stesso codice compilato senza OpenMP: e\x27 da"
        print "  li\x27 che va misurato lo speedup, non dalla colonna T=1."
    }' "$STUDY_CSV"
fi

study_end
