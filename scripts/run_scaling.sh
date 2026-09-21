#!/usr/bin/env bash
#
# Measurement of how much parallel computing pays off.
#
# Two studies, which answer two different questions:
#
#   strong  at equal problem, how much does the time shorten by adding
#           processes? Each process receives a smaller slice.
#
#   weak    at equal work per process, does the time stay the same while the
#           problem grows? For a stencil code like this it is the most honest
#           measurement: the real limit is the memory bandwidth, and strong
#           scaling saturates it quickly on a single machine.
#
#   hybrid  at equal occupied cores, is it better to spend them on processes
#           or on threads? The question is not idle: the processes divide the
#           domain, and as soon as a direction is divided the SIMD kernels
#           disappear and the Schur complement goes from one local solve to
#           three. The threads divide the lines, which are independent anyway,
#           so they pay neither the one nor the other.
#
# Every row contains the number of processes, the shape of the process grid,
# the global grid, the time per step and the share spent inside MPI. The
# results end up in build/scaling/results.csv.

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$root/build/scaling"
results="$build_dir/results${RESULTS_SUFFIX:-}.csv"
executable="$build_dir/paper_man"
build_log="$build_dir/build.log"
steps="${STEPS:-20}"
# On a laptop the background noise is high: every case is repeated and the best
# time is kept, which is the one least contaminated by other activities.
repeats="${REPEATS:-3}"
# SIMD=0 disables the vectorised kernels: it serves to compare on equal terms,
# since those apply only on the directions that are not divided.
simd="${SIMD:-1}"
# Which backend stitches the lines divided among the processes: schur or
# pipeline. The solver is no longer the flat glob src/*.c -- the backend lives
# in src/tridiag/$(TRIDIAG)/ -- and the flags are known only to the Makefile,
# so one compiles through it, as scripts/study/lib.sh does.
#
#   TRIDIAG=pipeline RESULTS_SUFFIX=_pipeline ./scripts/run_scaling.sh
tridiag="${TRIDIAG:-schur}"

# processes : shape of the process grid : global grid
strong_configs=(
    "1 : 1 1 1 : 128 128 128"
    "2 : 1 1 2 : 128 128 128"
    "4 : 1 2 2 : 128 128 128"
    "8 : 2 2 2 : 128 128 128"
)

# 64^3 cells per process in all the cases
weak_configs=(
    "1 : 1 1 1 : 64 64 64"
    "2 : 1 1 2 : 64 64 128"
    "4 : 1 2 2 : 64 128 128"
    "8 : 2 2 2 : 128 128 128"
)

# Same grid and same number of cores, shared out differently between processes
# and threads. The rows with a single process keep every direction whole: SIMD
# stays alive and Schur has no interfaces to stitch.
hybrid_configs=(
    "1 x 1 : 1 1 1 : 128 128 128"
    "1 x 2 : 1 1 1 : 128 128 128"
    "2 x 1 : 1 1 2 : 128 128 128"
    "1 x 4 : 1 1 1 : 128 128 128"
    "2 x 2 : 1 1 2 : 128 128 128"
    "4 x 1 : 1 2 2 : 128 128 128"
    "1 x 8 : 1 1 1 : 128 128 128"
    "2 x 4 : 1 1 2 : 128 128 128"
    "4 x 2 : 1 2 2 : 128 128 128"
    "8 x 1 : 2 2 2 : 128 128 128"
)

mkdir -p "$build_dir"
printf 'backend %s, SIMD=%s, %s steps, %s repeats\n' \
    "$tridiag" "$simd" "$steps" "$repeats"
trap 'rm -f "$executable"' EXIT
printf '%s\n' 'study,procs,threads,px,py,pz,nx,ny,nz,steps,wall_ms,mpi_ms' \
    > "$results"

run_case()
{
    local study="$1" procs="$2" threads="$3" shape="$4" grid="$5"
    read -r px py pz <<< "$shape"
    read -r nx ny nz <<< "$grid"

    printf 'Running %-6s procs=%-2s threads=%-2s shape=%sx%sx%s grid=%sx%sx%s\n' \
        "$study" "$procs" "$threads" "$px" "$py" "$pz" "$nx" "$ny" "$nz"

    # The threaded branch is compiled only when needed: the OpenMP build costs
    # one percentage point even with a single thread, and the two measurements
    # must be kept separate.
    local omp=0
    [[ "$threads" -gt 1 ]] && omp=1

    if ! make -s -B --no-print-directory -C "$root" \
            TRIDIAG="$tridiag" MPI=1 OMP="$omp" SIMD="$simd" \
            EXTRA_CPPFLAGS="-DDEFAULT_WIDTH=$nx -DDEFAULT_HEIGHT=$ny \
                -DDEFAULT_DEPTH=$nz \
                -DDEFAULT_T=1e-1 -DDEFAULT_STEPS=$steps" \
            build/tests/paper_man > "$build_log" 2>&1; then
        echo "build failed, see $build_log" >&2
        sed 's/^/    /' "$build_log" >&2
        exit 1
    fi
    mv "$root/build/tests/paper_man" "$executable"

    # --bind-to none is mandatory, not a detail: by default mpirun pins every
    # process to a single core, and the threads of that process share it out
    # instead of taking one each.  Without this option the threads column
    # measures zero gain and the measurement says nothing.
    local best_wall="" best_mpi="" output wall mpi
    for ((run = 0; run < repeats; run++)); do
        output="$(OMP_NUM_THREADS="$threads" \
                  mpirun --oversubscribe --bind-to none \
                  -n "$procs" "$executable" \
                  "$px" "$py" "$pz")"
        wall="$(awk '/wall per step:/ {print $4}' <<< "$output")"
        mpi="$(awk '/mpi per step:/  {print $4}' <<< "$output")"

        if [[ -z "$best_wall" ]] || \
           awk "BEGIN { exit !($wall < $best_wall) }"; then
            best_wall="$wall"
            best_mpi="$mpi"
        fi
    done
    printf '    best of %s: %s ms\n' "$repeats" "$best_wall"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$study" "$procs" "$threads" "$px" "$py" "$pz" \
        "$nx" "$ny" "$nz" "$steps" "$best_wall" "$best_mpi" >> "$results"
}

for study in strong weak; do
    declare -n configs="${study}_configs"
    for config in "${configs[@]}"; do
        IFS=':' read -r procs shape grid <<< "$config"
        run_case "$study" "${procs// /}" 1 "$(echo "$shape")" "$(echo "$grid")"
    done
done

# The hybrid study is skipped with HYBRID=0: without OpenMP in the compiler its
# rows cannot be compiled.
if [[ "${HYBRID:-1}" != "0" ]]; then
    for config in "${hybrid_configs[@]}"; do
        IFS=':' read -r spec shape grid <<< "$config"
        read -r procs _ threads <<< "$spec"
        run_case hybrid "$procs" "$threads" \
            "$(echo "$shape")" "$(echo "$grid")"
    done
fi

printf '\nResults in %s\n\n' "$results"

# In strong scaling the time should halve when the processes double, so the
# efficiency is (t1/tP)/P. In weak scaling the work per process does not
# change, so the time should stay constant and the efficiency is t1/tP.
awk -F, '
NR == 1 { next }
{
    if ($1 != study) {
        study = $1
        base = $11
        printf "\n%s scaling\n", study
        printf "  %-9s %-9s %-13s %11s %11s %9s %9s\n",
               "proc x th", "shape", "grid", "wall/step", "mpi/step",
               (study == "weak" ? "t1/tP" : "speedup"), "effic."
    }
    cores = $2 * $3
    ratio = base / $11
    efficiency = (study == "weak") ? ratio : ratio / cores
    printf "  %-9s %-9s %-13s %8.1f ms %8.1f ms %9.2f %8.0f%%\n",
           $2 " x " $3, $4 "x" $5 "x" $6, $7 "x" $8 "x" $9, $11, $12,
           ratio, 100 * efficiency
}
' "$results"
