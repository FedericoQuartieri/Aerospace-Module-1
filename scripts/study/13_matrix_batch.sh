#!/usr/bin/env bash
#PBS -N nsb-13-batch
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Fase 13 -- il batch della pipeline, contro tutto il resto.
#
# PIPELINE_BATCH_LINES e' l'unica manopola del backend pipeline, e serve a due
# cose che tirano in direzioni opposte:
#
#   fra processi   batch piccoli riempiono la pipeline prima, quindi meno
#                  tempo morto all'inizio, ma mandano piu' messaggi.
#   fra thread     dentro un batch le linee sono indipendenti e i thread se le
#                  spartiscono, ma ogni livello e' una barriera: batch piccoli
#                  vuol dire poco lavoro fra una barriera e la successiva.
#                  Misurato in locale: da 64 a 1024 il guadagno a 4 thread
#                  passa da 1.29x a 1.76x.
#
# Quindi il batch ottimo NON e' un numero, e' una funzione di quanti rank,
# quanti thread e quanto e' grande il blocco locale. Questa fase la campiona:
# dieci batch per sette piazzamenti per due taglie.
#
# Le righe schur sono il riferimento: allo stesso piazzamento, quanto fa
# l'altro backend, che di manopole non ne ha.
#
#   qsub scripts/study/13_matrix_batch.sh
#   BATCHES="64 1024" qsub scripts/study/13_matrix_batch.sh

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub va fatto dalla radice del repo; qui sono in $PWD" >&2
    exit 1
fi

source scripts/study/lib.sh
source scripts/study/matrix.sh

STUDY_CHAIN_MAX="$MATRIX_CHAIN_MAX"
study_begin 13_matrix_batch
study_machine

GRIDS="${GRIDS:-128 224}"
BATCHES="${BATCHES:-8 16 32 64 128 256 512 1024 2048 4096}"
# rank x thread: le colonne del rettangolo della 12 che contano qui. Il primo
# numero sono i rank, il secondo i thread.
PLACEMENTS="${PLACEMENTS:-1x1 1x14 1x28 1x56 8x1 8x7 28x2 56x1}"

CASE_BACKEND=pipeline
CASE_OMP=1
CASE_MPI=1
CASE_REPEATS="${REPEATS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-1200}"

for n in $GRIDS; do
    steps="$(matrix_steps "$n")"
    grid="$n $n $n"

    for simd in $MATRIX_SIMD; do
        echo "=== ${n}^3 simd=$simd: il batch, per ogni piazzamento ==="
        for place in $PLACEMENTS; do
            r="${place%x*}"
            t="${place#*x}"
            [[ $(( r * t )) -le "$(matrix_units)" ]] || continue
            shape="$(study_auto_shape "$r")"
            [[ -n "$shape" ]] || continue
            matrix_shape_fits "$shape" "$grid" || continue

            for b in $BATCHES; do
                study_case label="pipeline ${n}^3 s$simd $place b=$b" \
                    backend=pipeline batch="$b" simd="$simd" \
                    ranks="$r" threads="$t" shape="$shape" \
                    grid="$grid" steps="$steps"
            done

            # Il riferimento senza manopole, allo stesso piazzamento.
            study_case label="schur ${n}^3 s$simd $place" \
                backend=schur simd="$simd" ranks="$r" threads="$t" \
                shape="$shape" grid="$grid" steps="$steps" \
                note="riferimento per il batch"
        done
        echo
    done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== il batch migliore, per piazzamento ==="
    awk -F, -v phase=13_matrix_batch '
    NR == 1 || $1 != phase || $33 != "ok" { next }
    {
        k = $10 "," $5 "," $8 "x" $9
        if ($3 == "schur") { ref[k] = $17 + 0; next }
        if (!(k in best) || $17 + 0 < best[k]) { best[k] = $17 + 0; bb[k] = $4 }
        if (!(k in worst) || $17 + 0 > worst[k]) { worst[k] = $17 + 0; wb[k] = $4 }
        if (!(k in seen)) { keys[++nk] = k; seen[k] = 1 }
    }
    END {
        printf "  %-7s %-5s %-8s %9s %7s %9s %7s %7s %9s\n",
               "griglia", "simd", "rankxthr", "migliore", "batch",
               "peggiore", "batch", "scarto", "schur"
        for (i = 1; i <= nk; i++) {
            split(keys[i], p, ",")
            printf "  %-7s %-5s %-8s %9.1f %7s %9.1f %7s %6.2fx %9s\n",
                   p[1], p[2], p[3],
                   best[keys[i]], bb[keys[i]], worst[keys[i]], wb[keys[i]],
                   (best[keys[i]] > 0 ? worst[keys[i]] / best[keys[i]] : 0),
                   (keys[i] in ref ? sprintf("%.1f", ref[keys[i]]) : "-")
        }
        print ""
        print "  Se la colonna `batch migliore\x27 cambia con il piazzamento, allora"
        print "  il default 64 e\x27 giusto solo per il piazzamento su cui fu scelto."
    }' "$STUDY_CSV"
fi

study_end
