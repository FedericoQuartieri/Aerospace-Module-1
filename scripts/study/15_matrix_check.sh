#!/usr/bin/env bash
#PBS -N nsb-15-check
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Phase 15 -- does the matrix still solve the right problem?
#
# The other five phases measure times. A time of a configuration that gives the
# wrong answer is not a slow time, it is a number without meaning, and the
# campaign produces thousands of them without looking even once at the result.
# This phase looks only at that.
#
# Every case runs with BENCH_NORMS=1, which computes the error norms against
# the exact solution, and puts them in the CSV (columns l2_ux and l2_p). The
# property to verify is that they are THE SAME for every configuration: the
# result must not depend on how many processes, how many threads, which
# backend, which shape of the grid, which batch. It is the same property that
# check_pipeline.sh verifies on a small scale, here widened to the whole
# matrix.
#
# Small grid on purpose: correctness does not need the size, and at 64^3 these
# cases cost a few seconds each. The column that matters is the difference
# between the norms, not their value.
#
# A case that comes out different from the others here invalidates all the rows
# of the other phases with the same configuration, and must be solved before
# reading any time.
#
#   qsub scripts/study/15_matrix_check.sh

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub must be run from the repo root; I am in $PWD" >&2
    exit 1
fi

source scripts/study/lib.sh
source scripts/study/matrix.sh

STUDY_CHAIN_MAX="$MATRIX_CHAIN_MAX"
study_begin 15_matrix_check
STUDY_EXPECTED="$STUDY_OUT/expected.tsv"
: > "$STUDY_EXPECTED"
study_machine

GRID="${CHECK_GRID:-64}"
CHECK_RANKS="${CHECK_RANKS:-1 2 4 7 8 14 28}"
CHECK_THREADS="${CHECK_THREADS:-1 2 7 8}"
# auto is the batch chosen at start-up, the one the other phases use: it must
# give the same norms as the values fixed at compile time.
CHECK_BATCHES="${CHECK_BATCHES:-1 7 64 1024 auto}"

grid="$GRID $GRID $GRID"
steps="$(matrix_steps "$GRID")"

CASE_MPI=1
CASE_OMP=1
# One repetition is enough: nothing is timed here.
CASE_REPEATS=1
CASE_NORMS=1
CASE_TIMEOUT="${CASE_TIMEOUT:-600}"

# The true reference of the whole phase. Here it does not only serve to
# normalise the times: it is the L2 norm against which all the following rows
# must match. A binary without MPI and without OpenMP cannot go wrong because
# of a division or a thread, so if a shape departs from it, it is that shape
# that is wrong, not the reference.
echo "=== the baselines: serial and single process ==="
# Always present, even when MATRIX_BACKENDS/SIMD restrict the sweep.
study_case label="reference serial" backend=schur simd=0 omp=0 mpi=0 \
    ranks=1 threads=1 shape="1 1 1" grid="$grid" steps="$steps"
for backend in $MATRIX_BACKENDS; do
    for simd in $MATRIX_SIMD; do
        study_baseline label="$backend s$simd" \
            backend="$backend" simd="$simd" grid="$grid" steps="$steps"
    done
done
for simd in $MATRIX_SIMD; do
    for b in $CHECK_BATCHES; do
        study_baseline label="pipeline b=$b s$simd" \
            backend=pipeline batch="$b" simd="$simd" \
            grid="$grid" steps="$steps"
    done
done
echo

echo "=== ${GRID}^3: every process-grid shape, one thread ==="
for backend in $MATRIX_BACKENDS; do
    for simd in $MATRIX_SIMD; do
        for n in $CHECK_RANKS; do
            forme=()
            mapfile -t forme < <(matrix_shapes "$n")
            for shape in "${forme[@]}"; do
                matrix_shape_fits "$shape" "$grid" || continue
                study_case label="$backend R=$n ${shape// /x} s$simd" \
                    backend="$backend" simd="$simd" ranks="$n" threads=1 \
                    shape="$shape" grid="$grid" steps="$steps"
            done
        done
    done
done
echo

echo "=== ${GRID}^3: threads must not change the result ==="
for backend in $MATRIX_BACKENDS; do
    for simd in $MATRIX_SIMD; do
        for t in $CHECK_THREADS; do
            for n in 1 4 8; do
                [[ $(( n * t )) -le "$(matrix_units)" ]] || continue
                shape="$(study_auto_shape "$n")"
                [[ -n "$shape" ]] || continue
                study_case label="$backend R=$n T=$t s$simd" \
                    backend="$backend" simd="$simd" ranks="$n" threads="$t" \
                    shape="$shape" grid="$grid" steps="$steps"
            done
        done
    done
done
echo

echo "=== ${GRID}^3: nor must the pipeline batch change it ==="
for simd in $MATRIX_SIMD; do
    for b in $CHECK_BATCHES; do
        for n in 1 8; do
            shape="$(study_auto_shape "$n")"
            study_case label="pipeline R=$n b=$b s$simd" \
                backend=pipeline batch="$b" simd="$simd" ranks="$n" \
                threads=1 shape="$shape" grid="$grid" steps="$steps"
        done
    done
done

verdict=0
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    if [[ "$STUDY_PENDING" -gt 0 ]]; then
        echo "Numerical verdict pending: the campaign still has unfinished cases."
    elif ! python3 "$STUDY_ROOT/scripts/study/validate.py" \
            "$STUDY_CSV" "$STUDY_EXPECTED" "${CHECK_TOLERANCE:-1e-10}"; then
        verdict=3
    fi
fi

study_end
exit "$verdict"
