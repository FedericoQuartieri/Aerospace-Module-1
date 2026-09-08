#!/usr/bin/env bash

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$root/build/pipeline-check"
grid="${GRID:-16}"
steps="${STEPS:-4}"
t_end="${T_END:-1.0}"
wr_freq="${WR_FREQ:-9999}"
batches="${PIPELINE_BATCHES:-1 3 64}"
mpi="${MPI:-0}"
omp="${OMP:-0}"
simd="${SIMD:-0}"
ranks="${RANKS:-1}"

tests=(paper_man zero_pressure constant_forcing_man)
targets=()
for test in "${tests[@]}"; do
    targets+=("build/tests/$test")
done

extra_cppflags="-DDEFAULT_WIDTH=$grid -DDEFAULT_HEIGHT=$grid"
extra_cppflags+=" -DDEFAULT_DEPTH=$grid -DDEFAULT_STEPS=$steps"
extra_cppflags+=" -DDEFAULT_T=$t_end -DDEFAULT_WR_FREQ=$wr_freq"

mkdir -p "$build_dir"

run_test()
{
    local test="$1"
    local out="$2"

    if [[ "$mpi" == "1" ]]; then
        "${MPIRUN:-mpirun}" -n "$ranks" "$root/build/tests/$test" > "$out"
    else
        "$root/build/tests/$test" > "$out"
    fi
}

build_backend()
{
    local backend="$1"
    local batch="$2"

    make -B -C "$root" TRIDIAG="$backend" PIPELINE_BATCH_LINES="$batch" \
        MPI="$mpi" OMP="$omp" SIMD="$simd" \
        EXTRA_CPPFLAGS="$extra_cppflags" "${targets[@]}"
}

capture_suite()
{
    local label="$1"

    for test in "${tests[@]}"; do
        local out="$build_dir/$label.$test.out"
        local norms="$build_dir/$label.$test.norms"

        run_test "$test" "$out"
        grep '^  L2 error' "$out" > "$norms"
    done
}

printf '=== reference: Schur, grid %sx%sx%s, steps %s ===\n' \
    "$grid" "$grid" "$grid" "$steps"
build_backend schur 64
capture_suite schur

for batch in $batches; do
    label="pipeline-batch-$batch"

    printf '\n=== pipeline: batch %s ===\n' "$batch"
    build_backend pipeline "$batch"
    capture_suite "$label"

    for test in "${tests[@]}"; do
        diff -u "$build_dir/schur.$test.norms" \
                "$build_dir/$label.$test.norms"
    done
done

printf '\nPASSED: pipeline norms match Schur for batches: %s\n' "$batches"
