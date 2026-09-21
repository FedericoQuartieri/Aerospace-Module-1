#!/usr/bin/env bash
#PBS -N nsb-13-batch
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Phase 13 -- the pipeline batch, against everything else.
#
# PIPELINE_BATCH_LINES is the only knob of the pipeline backend, and it serves
# two things that pull in opposite directions:
#
#   among processes   small batches fill the pipeline sooner, so less dead time
#                     at the start, but they send more messages.
#   among threads     inside a batch the lines are independent and the threads
#                     share them out, one whole line each. Up to 46146d6 every
#                     level was a barrier and small batches left the threads
#                     almost nothing to do between two waits; now the barrier
#                     is one per batch, and the optimum must be found again.
#
# On an axis that no process divides the pipeline cuts the batch to one per
# worker, so with a single process the batch changes little more than the work
# memory. To know which batch many threads want, placements are needed with
# more processes and many threads each: 14x4, 8x7, 4x14, 2x28.
#
# So the optimal batch is NOT a number, it is a function of how many ranks, how
# many threads and how large the local block is. This phase samples it: eleven
# batches for ten placements for two sizes. The summary at the bottom compares
# the best with the one given by the rule used at start-up.
#
# The schur rows are the reference: at the same placement, what the other
# backend does, which has no knobs.
#
#   qsub scripts/study/13_matrix_batch.sh
#   BATCHES="64 1024" qsub scripts/study/13_matrix_batch.sh

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub must be run from the repo root; I am in $PWD" >&2
    exit 1
fi

source scripts/study/lib.sh
source scripts/study/matrix.sh

STUDY_CHAIN_MAX="$MATRIX_CHAIN_MAX"
study_begin 13_matrix_batch
study_machine

GRIDS="${GRIDS:-128 224}"
# 8192 beyond the ceiling of the rule (4096): without it, a minimum on the edge
# does not say whether the ceiling costs anything.
BATCHES="${BATCHES:-8 16 32 64 128 256 512 1024 2048 4096 8192}"
# rank x thread: the columns of the rectangle of phase 12 that matter here. The
# first number is the ranks, the second the threads.
PLACEMENTS="${PLACEMENTS:-1x1 1x14 1x56 8x1 56x1 14x4 28x2 8x7 4x14 2x28}"

CASE_BACKEND=pipeline
CASE_OMP=1
CASE_MPI=1
CASE_REPEATS="${REPEATS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-1200}"

for n in $GRIDS; do
    steps="$(matrix_steps "$n")"
    grid="$n $n $n"

    for simd in $MATRIX_SIMD; do
        # The references of the size. The batch enters here too: in the
        # pipeline it decides the scratch memory and the way of walking it, so
        # one serial per batch is not a duplicate. Schur has only one, it does
        # not even look at the batch.
        for b in $BATCHES; do
            study_baseline label="pipeline ${n}^3 s$simd b=$b" \
                backend=pipeline batch="$b" simd="$simd" \
                grid="$grid" steps="$steps"
        done
        study_baseline label="schur ${n}^3 s$simd" \
            backend=schur simd="$simd" grid="$grid" steps="$steps" \
            note="riferimento per il batch"

        echo "=== ${n}^3 simd=$simd: the batch, for every placement ==="
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

            # The reference without knobs, at the same placement.
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
    echo "=== the best batch, per placement ==="
    awk -F, -v phase=13_matrix_batch '
    # The same rule as batch_lines_for_threads in
    # src/tridiag/pipeline/backend.c: if it changes there, it must be changed
    # here.
    function rule(t,    target, power) {
        target = 256 * (t > 0 ? t : 1)
        power = 1
        while (power * 2 <= target) power *= 2
        if (target * target > 2 * power * power) power *= 2
        return power < 4096 ? power : 4096
    }
    NR == 1 || $1 != phase || $(NF - 1) != "ok" { next }
    # As above: the references collide with the 1x1 placement.
    $2 ~ / (seriale|T\(1\))$/ { next }
    {
        k = $10 "," $5 "," $8 "x" $9
        if ($3 == "schur") { ref[k] = $17 + 0; next }
        at[k, $4] = $17 + 0
        if (!(k in best) || $17 + 0 < best[k]) { best[k] = $17 + 0; bb[k] = $4 }
        if (!(k in worst) || $17 + 0 > worst[k]) { worst[k] = $17 + 0; wb[k] = $4 }
        if (!(k in seen)) { keys[++nk] = k; seen[k] = 1 }
    }
    END {
        printf "  %-7s %-5s %-8s %9s %6s %9s %6s %7s %6s %7s %9s\n",
               "grid", "simd", "rankxthr", "best", "batch",
               "worst", "batch", "spread", "rule", "loss", "schur"
        for (i = 1; i <= nk; i++) {
            k = keys[i]
            split(k, p, ",")
            split(p[3], rt, "x")
            r = rule(rt[2])
            loss = "-"
            if (((k, r) in at) && best[k] > 0)
                loss = sprintf("%+.1f%%", 100 * (at[k, r] / best[k] - 1))
            printf "  %-7s %-5s %-8s %9.1f %6s %9.1f %6s %6.2fx %6s %7s %9s\n",
                   p[1], p[2], p[3],
                   best[k], bb[k], worst[k], wb[k],
                   (best[k] > 0 ? worst[k] / best[k] : 0), r, loss,
                   (k in ref ? sprintf("%.1f", ref[k]) : "-")
        }
        print ""
        print "  rule is the batch the solver picks at start-up for that many"
        print "  threads per process, loss how much slower it is than the best."
        print "  A large loss means the rule in backend.c has to change."
    }' "$STUDY_CSV"
fi

study_end
