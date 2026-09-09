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
tolerance="${TOLERANCE:-1e-10}"
process_grid="${PROCESS_GRID:-}"
proc_args=()

if [[ -n "$process_grid" ]]; then
    read -r px py pz extra <<< "$process_grid"
    if [[ -z "${px:-}" || -z "${py:-}" || -z "${pz:-}" ||
          -n "${extra:-}" ]]; then
        echo "PROCESS_GRID must contain exactly three integers, e.g. '1 2 2'" >&2
        exit 2
    fi
    proc_args=("$px" "$py" "$pz")
fi

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
        if [[ -n "$process_grid" ]]; then
            "${MPIRUN:-mpirun}" -n "$ranks" \
                "$root/build/tests/$test" "${proc_args[@]}" > "$out"
        else
            "${MPIRUN:-mpirun}" -n "$ranks" \
                "$root/build/tests/$test" > "$out"
        fi
    else
        if [[ -n "$process_grid" ]]; then
            "$root/build/tests/$test" "${proc_args[@]}" > "$out"
        else
            "$root/build/tests/$test" > "$out"
        fi
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

compare_norms()
{
    local reference="$1"
    local candidate="$2"
    local label="$3"
    local test="$4"

    awk -v tol="$tolerance" -v label="$label" -v test="$test" '
    function abs(x) { return x < 0 ? -x : x }
    function fail(message) {
        print message > "/dev/stderr"
        failed = 1
    }
    NR == FNR {
        ref[++n] = $NF + 0
        next
    }
    {
        got[++m] = $NF + 0
    }
    END {
        if (n == 0 || m == 0) {
            fail(label " " test ": missing L2 error lines")
        }
        if (n != m) {
            fail(label " " test ": different number of L2 error lines")
        }
        limit = n < m ? n : m
        for (i = 1; i <= limit; i++) {
            scale = abs(ref[i])
            if (abs(got[i]) > scale) {
                scale = abs(got[i])
            }
            if (scale < 1) {
                scale = 1
            }
            diff = abs(got[i] - ref[i])
            if (diff > tol * scale) {
                printf "%s %s: norm %d differs: reference %.17g, candidate %.17g, diff %.3e, tolerance %.3e\n", \
                    label, test, i, ref[i], got[i], diff, tol * scale \
                    > "/dev/stderr"
                failed = 1
            }
        }
        exit failed ? 1 : 0
    }' "$reference" "$candidate"
}

printf '=== reference: Schur, grid %sx%sx%s, steps %s ===\n' \
    "$grid" "$grid" "$grid" "$steps"
printf '    mpi=%s ranks=%s process_grid=%s tolerance=%s\n' \
    "$mpi" "$ranks" "${process_grid:-auto}" "$tolerance"
build_backend schur 64
capture_suite schur

for batch in $batches; do
    label="pipeline-batch-$batch"

    printf '\n=== pipeline: batch %s ===\n' "$batch"
    build_backend pipeline "$batch"
    capture_suite "$label"

    for test in "${tests[@]}"; do
        compare_norms "$build_dir/schur.$test.norms" \
            "$build_dir/$label.$test.norms" "$label" "$test"
    done
done

printf '\nPASSED: pipeline norms match Schur for batches: %s\n' "$batches"
