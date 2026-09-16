#!/usr/bin/env bash
#
# One problem under every way the solver can be built: both backends, one
# process or the grid split along every axis, with and without threads and
# SIMD.  The L2 norms of test/paper_man come out side by side, with the
# largest relative difference from the serial Schur run.
#
#   ./scripts/run_equivalence.sh
#   RANKS=8 PROCESS_GRID="2 2 2" THREADS=2 ./scripts/run_equivalence.sh
#
# The run measures results, not times, so oversubscribing the machine is
# harmless: 8 processes with 2 threads each on a laptop give the same digits,
# only later.  Passive waiting keeps the idle threads from spinning.
#
# Results go to docs/equivalence/, next to a log of commit and host.

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${OUT_DIR:-$root/docs/equivalence}"
grid="${GRID:-32}"
steps="${STEPS:-20}"
t_end="${T_END:-0.1}"
ranks="${RANKS:-8}"
process_grid="${PROCESS_GRID:-2 2 2}"
threads="${THREADS:-2}"
read -r -a mpirun <<< "${MPIRUN:-mpirun --oversubscribe}"
read -r -a proc_args <<< "$process_grid"

mkdir -p "$out_dir"
results="$out_dir/results.csv"
log="$out_dir/run.log"

{
    printf 'commit %s%s\n' "$(git -C "$root" rev-parse --short HEAD)" \
        "$(git -C "$root" diff --quiet HEAD -- src include test Makefile ||
           echo ' (with uncommitted changes)')"
    printf 'host %s, %s\n' "$(hostname)" "$(date -Is)"
    printf 'paper_man at %s^3, %s steps to T = %s; MPI runs: %s processes (%s), %s threads\n\n' \
        "$grid" "$steps" "$t_end" "$ranks" "$process_grid" "$threads"
} > "$log"

extra="-DDEFAULT_WIDTH=$grid -DDEFAULT_HEIGHT=$grid -DDEFAULT_DEPTH=$grid"
extra+=" -DDEFAULT_STEPS=$steps -DDEFAULT_T=$t_end"

printf 'backend,processes,threads,simd,L2_ux,L2_uy,L2_uz,L2_p\n' > "$results"

for backend in schur pipeline; do
    for mpi in 0 1; do
        for omp in 0 1; do
            for simd in 0 1; do
                label="$backend-mpi$mpi-omp$omp-simd$simd"
                bin_dir="build/equivalence/$label"

                if ! make -s -B --no-print-directory -C "$root" \
                        TRIDIAG="$backend" MPI="$mpi" OMP="$omp" SIMD="$simd" \
                        EXTRA_CPPFLAGS="$extra" TEST_BIN_DIR="$bin_dir" \
                        "$bin_dir/paper_man" >> "$log" 2>&1; then
                    echo "build $label failed, see $log" >&2
                    exit 1
                fi

                procs=1
                nthreads=1
                command=("$root/$bin_dir/paper_man")
                if [[ $mpi == 1 ]]; then
                    procs="$ranks"
                    command=("${mpirun[@]}" -n "$ranks" "${command[@]}" "${proc_args[@]}")
                fi
                if [[ $omp == 1 ]]; then
                    nthreads="$threads"
                fi

                output="$(OMP_NUM_THREADS="$nthreads" OMP_WAIT_POLICY=passive \
                          "${command[@]}")"
                printf '=== %s\n%s\n' "$label" "$output" >> "$log"

                printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
                    "$backend" "$procs" "$nthreads" "$simd" \
                    "$(awk '/L2 error u_x:/ {print $NF}' <<< "$output")" \
                    "$(awk '/L2 error u_y:/ {print $NF}' <<< "$output")" \
                    "$(awk '/L2 error u_z:/ {print $NF}' <<< "$output")" \
                    "$(awk '/L2 error p:/   {print $NF}' <<< "$output")" \
                    >> "$results"
                printf '  %-28s done\n' "$label"
            done
        done
    done
done

awk -F, -v tolerance="${TOLERANCE:-1e-10}" '
function abs(v) { return v < 0 ? -v : v }
BEGIN {
    if (tolerance !~ /^[0-9.]+([eE][-+]?[0-9]+)?$/ || tolerance + 0 <= 0) {
        print "invalid TOLERANCE"; failed = 1; exit 3
    }
}
NR == 1 {
    printf "\n%-9s %5s %7s %4s  %-17s %-17s %-17s %-17s %s\n",
           "backend", "procs", "threads", "simd", "L2 u_x", "L2 u_y", "L2 u_z",
           "L2 p", "max scaled diff"
    next
}
NR == 2 { for (c = 5; c <= 8; c++) ref[c] = $c }
{
    count++
    worst = 0
    for (c = 5; c <= 8; c++) {
        if ($c !~ /^[-+]?[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$/ ||
            $c + 0 < 0 || $c + 0 > 1e300) {
            printf "invalid norm in %s, column %d\n", $1, c
            failed = 1
        }
        scale = abs(ref[c]); if (scale < 1) scale = 1
        d = abs($c - ref[c]) / scale
        if (d > worst) worst = d
    }
    if (worst > tolerance) failed = 1
    printf "%-9s %5s %7s %4s  %-17s %-17s %-17s %-17s %.1e\n",
           $1, $2, $3, $4, $5, $6, $7, $8, worst
}
END {
    if (count != 16) failed = 1
    print failed ? "FAILED: incomplete or different results" : "PASSED: all 16 configurations agree"
    exit failed ? 3 : 0
}' "$results" | tee -a "$log"

printf '\nresults in %s\nlog in %s\n' "$results" "$log"
