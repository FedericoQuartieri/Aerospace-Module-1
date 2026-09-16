#!/usr/bin/env bash
#
# Error of test/brinkman_channel against its exact solution, over grids and
# permeabilities, for the uniform and the porous-layer case.
#
#   ./scripts/run_brinkman.sh
#   KS="1e-2 1e-4" GRIDS="32 64 128" ./scripts/run_brinkman.sh
#
# The flow depends on y only, so the grid is refined along Y and kept at a
# few cells along X and Z: the whole study runs in a few minutes on a laptop.
# One binary serves every case, the grid comes from a configuration file.
#
# Results go to docs/brinkman/ rather than build/, which is not tracked: a
# figure without the numbers behind it cannot be checked or extended.
#
# Past K = 1e-4 at dt = 1/200, Crank-Nicolson barely damps the drag inside
# the solid (the factor is (1 - a) / (1 + a) with a = dt NU / (2 K)), and the
# error has not settled by T = 1.  That is a property of the time scheme, not
# of the spatial error this study measures, so the default stops at 1e-4.

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$root/build/brinkman"
out_dir="${OUT_DIR:-$root/docs/brinkman}"
executable="$build_dir/brinkman_channel"

cases="${CASES:-uniform layer}"
ks="${KS:-1 1e-1 1e-2 1e-3 1e-4}"
grids="${GRIDS:-16 32 64 128 256 512}"
steps="${STEPS:-200}"
t_end="${T_END:-1.0}"

mkdir -p "$build_dir" "$out_dir"
raw="$build_dir/raw.csv"
results="$out_dir/results.csv"
log="$out_dir/run.log"

{
    printf 'commit %s%s\n' "$(git -C "$root" rev-parse --short HEAD)" \
        "$(git -C "$root" diff --quiet HEAD -- src include test Makefile ||
           echo ' (with uncommitted changes)')"
    printf 'host %s, %s\n' "$(hostname)" "$(date -Is)"
    printf 'steps %s, t_end %s\n\n' "$steps" "$t_end"
} > "$log"

if ! make -s -B --no-print-directory -C "$root" \
        MPI=0 OMP=0 SIMD=0 TRIDIAG=schur TEST_BIN_DIR=build/brinkman \
        build/brinkman/brinkman_channel >> "$log" 2>&1; then
    echo "build failed, see $log" >&2
    exit 1
fi

printf 'case,K,N,dy,sqrtK_over_dy,rel_l2,linf_ux,linf_uyuz,porous_max_num,porous_max_exact,status\n' \
    > "$raw"

for case in $cases; do
    for k in $ks; do
        for n in $grids; do
            config="$build_dir/grid-$n.txt"
            printf 'width = 6\nheight = %s\ndepth = 6\nlx = 20\nly = 1\nlz = 20\nt_end = %s\nsteps = %s\n' \
                "$n" "$t_end" "$steps" > "$config"

            status=ok
            output="$("$executable" "$case" "$k" "$config")" || {
                code=$?
                # 1 is the test's threshold: the grid does not resolve this
                # K, which is a result.  Anything else is a crash.
                if [[ $code -ne 1 ]]; then
                    printf '%s\n' "$output" >&2
                    echo "brinkman_channel $case $k (N=$n) exited with $code" >&2
                    exit 1
                fi
                status=over-threshold
            }
            printf '%s\n' "$output" >> "$log"

            porous="$(awk '/porous max/ { gsub(",", ""); print $(NF-2) "," $NF }' \
                      <<< "$output")"
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "$case" "$k" "$n" \
                "$(awk '/^  dy:/ {print $NF}' <<< "$output")" \
                "$(awk '/sqrt\(K\) \/ dy:/ {print $NF}' <<< "$output")" \
                "$(awk '/relative L2 error u:/ {print $NF}' <<< "$output")" \
                "$(awk '/Linf error u_x:/ {print $NF}' <<< "$output")" \
                "$(awk '/Linf u_y, u_z:/ {print $NF}' <<< "$output")" \
                "${porous:-,}" "$status" >> "$raw"
            printf '  %-7s K=%-5s N=%-4s rel L2 %s  %s\n' "$case" "$k" "$n" \
                "$(awk '/relative L2 error u:/ {print $NF}' <<< "$output")" \
                "$status"
        done
    done
done

# Observed order between consecutive grids of the same case and K.
awk -F, -v OFS=, -v results="$results" '
NR == 1 { print $0, "rate" > results; next }
{
    key = $1 FS $2
    rate = "nan"
    if (key == previous_key && $6 > 0 && previous_error > 0) {
        rate = log(previous_error / $6) / log(previous_dy / $4)
    }
    print $0, rate > results
    previous_key = key
    previous_error = $6
    previous_dy = $4
}' "$raw"

printf '\nresults in %s\nlog in %s\n' "$results" "$log"
