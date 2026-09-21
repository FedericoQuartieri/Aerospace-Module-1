#!/usr/bin/env bash

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$root/build/convergence"
raw_file="$build_dir/.raw.csv"
results_file="$build_dir/results.csv"
executable="$build_dir/paper_man"
build_log="$build_dir/build.log"
# The solver is no longer the flat glob src/*.c: the tridiagonal backend lives
# in src/tridiag/$(TRIDIAG)/ and the flags are known only to the Makefile. From
# here only its variables are passed, as scripts/study/lib.sh does.
#
# TRIDIAG, SIMD and OMP come from the environment with the defaults of the
# Makefile. They are needed on the cluster, where without SIMD and without
# threads the 256^3 grid of the temporal study would not finish within any
# reasonable walltime:
#
#   SIMD=1 OMP=1 ./scripts/run_convergence.sh
#   TRIDIAG=pipeline ./scripts/run_convergence.sh
#
# EXTRA_CFLAGS and EXTRA_CPPFLAGS are passed to the Makefile as they are.
tridiag="${TRIDIAG:-schur}"
simd="${SIMD:-0}"
omp="${OMP:-0}"
extra_cflags="${EXTRA_CFLAGS:-}"
extra_cppflags="${EXTRA_CPPFLAGS:-}"

# Each configuration contains: grid size, dt, total time.
spatial_configs=(
    "32  1e-4 1e-3"
    "64  1e-4 1e-3"
    "128  1e-4 1e-3"
    "256 1e-4 1e-3"
)

temporal_configs=(
    "256 0.05    1.0"
    "256 0.025   1.0"
    "256 0.0125  1.0"
    "256 0.00625 1.0"
)

mkdir -p "$build_dir"
printf 'backend %s, SIMD=%s, OMP=%s\n' "$tridiag" "$simd" "$omp"
trap 'rm -f "$raw_file" "$executable"' EXIT
printf '%s\n' \
    'study,N,h,steps,dt,T,L2_ux,L2_uy,L2_uz,L2_p' > "$raw_file"

run_case()
{
    local study="$1"
    local grid="$2"
    local dt="$3"
    local total_time="$4"
    local steps
    local output
    local h

    steps="$(awk -v t="$total_time" -v dt="$dt" \
        'BEGIN { printf "%.0f", t / dt }')"

    printf 'Running %-8s N=%-3s DT=%-8s STEPS=%-3s\n' \
        "$study" "$grid" "$dt" "$steps"

    # The grid is a compile-time constant: the tests read no configuration
    # file.
    if ! make -s -B --no-print-directory -C "$root" \
            TRIDIAG="$tridiag" SIMD="$simd" OMP="$omp" MPI=0 \
            EXTRA_CFLAGS="$extra_cflags" \
            EXTRA_CPPFLAGS="$extra_cppflags \
                -DDEFAULT_WIDTH=$grid -DDEFAULT_HEIGHT=$grid \
                -DDEFAULT_DEPTH=$grid \
                -DDEFAULT_T=$total_time -DDEFAULT_STEPS=$steps" \
            build/tests/paper_man > "$build_log" 2>&1; then
        echo "build failed, see $build_log" >&2
        sed 's/^/    /' "$build_log" >&2
        exit 1
    fi
    mv "$root/build/tests/paper_man" "$executable"

    output="$("$executable")"
    h="$(awk -v n="$grid" \
        'BEGIN { printf "%.17g", atan2(0, -1) / (n - 0.5) }')"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$study" "$grid" "$h" "$steps" "$dt" "$total_time" \
        "$(awk '/L2 error u_x:/ {print $NF}' <<< "$output")" \
        "$(awk '/L2 error u_y:/ {print $NF}' <<< "$output")" \
        "$(awk '/L2 error u_z:/ {print $NF}' <<< "$output")" \
        "$(awk '/L2 error p:/   {print $NF}' <<< "$output")" \
        >> "$raw_file"
}

for config in "${spatial_configs[@]}"; do
    read -r grid dt total_time <<< "$config"
    run_case spatial "$grid" "$dt" "$total_time"
done

for config in "${temporal_configs[@]}"; do
    read -r grid dt total_time <<< "$config"
    run_case temporal "$grid" "$dt" "$total_time"
done

awk -F, -v OFS=',' -v results="$results_file" '
function rate(coarse_error, fine_error, coarse_h, fine_h) {
    return log(coarse_error / fine_error) / log(coarse_h / fine_h)
}
NR == 1 {
    print $0, "rate_ux", "rate_uy", "rate_uz", "rate_p" > results
    next
}
{
    scale = ($1 == "temporal") ? $5 : $3

    if ($1 != previous_study) {
        rate_ux = rate_uy = rate_uz = rate_p = "nan"
        printf "\n%s convergence rates:\n", $1
    } else {
        rate_ux = rate(previous_ux, $7, previous_scale, scale)
        rate_uy = rate(previous_uy, $8, previous_scale, scale)
        rate_uz = rate(previous_uz, $9, previous_scale, scale)
        rate_p  = rate(previous_p, $10, previous_scale, scale)
        printf "  ux=%.4f  uy=%.4f  uz=%.4f  p=%.4f\n",
               rate_ux, rate_uy, rate_uz, rate_p
    }

    print $0, rate_ux, rate_uy, rate_uz, rate_p > results
    previous_study = $1
    previous_scale = scale
    previous_ux = $7
    previous_uy = $8
    previous_uz = $9
    previous_p = $10
}
' "$raw_file"

printf '\nResults written to %s\n' "$results_file"
