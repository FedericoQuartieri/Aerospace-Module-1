#!/usr/bin/env bash
#PBS -N nsb-10-threads
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Phase 10 -- a single rank, increasing threads, for all the variants.
#
# Phase 02 asks whether the threads pay off. This one fills the row: the two
# backends, with and without SIMD, on three sizes, from one thread to all the
# logical cpus. It serves to separate three things that in the old measurements
# were together:
#
#   - how much each backend scales on its own, without a divided domain;
#   - how much of that scaling is SIMD and how much is threads, which do not
#     add up: the vector kernels saturate the bandwidth before the threads do;
#   - where it stops, and whether the point where it stops depends on the
#     size. On a stencil the ceiling is the memory bandwidth, and the
#     bandwidth per core gets worse with the size.
#
# The rows with OMP=0 are the reference: the threaded build costs about 1% even
# when the threads are one, and without those rows that 1% ends up inside the
# speedup.
#
# The tail of the placement: the same 28 threads on one socket or distributed
# over two are the same computing power with half or all of the memory
# channels. On a stencil this difference is often larger than that between two
# algorithms.
#
#   qsub scripts/study/10_matrix_threads.sh
#   GRIDS=128 qsub scripts/study/10_matrix_threads.sh      a single size

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub must be run from the repo root; I am in $PWD" >&2
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

    echo "=== ${n}^3, $steps steps: one rank, increasing threads ==="
    for backend in $MATRIX_BACKENDS; do
        for simd in $MATRIX_SIMD; do
            # The two references of the configuration: the true serial -- no
            # MPI linked, no OpenMP compiled -- and the same thing with the
            # parallel machinery inside but a single process and thread. The
            # speedups of this phase are normalised on them, and their
            # difference says how much that machinery costs when idle.
            study_baseline label="$backend ${n}^3 s$simd" \
                backend="$backend" simd="$simd" \
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

# The placement, at equal threads: `close' packs them onto the first socket as
# long as they fit, `spread' distributes them over all of them. Half of the
# memory channels against all.
BIND_GRID="${BIND_GRID:-224}"
BIND_THREADS="${BIND_THREADS:-14 28}"
echo "=== ${BIND_GRID}^3: the same threads, packed or spread ==="
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
    echo "=== result: ms/step and speedup against its own 1-thread case ==="
    # The list of threads comes from the shell: sorting it inside awk would
    # need asort, which belongs to gawk, and on the cluster awk may be mawk.
    awk -F, -v phase=10_matrix_threads -v tlist="$THREADS" '
    NR == 1 || $1 != phase || $(NF - 1) != "ok" || $2 ~ /bind=/ { next }
    {
        key = $3 "," $5 "," $10
        if (!(key in seen)) { keys[++nk] = key; seen[key] = 1 }
        # The two reference rows are not points of the curve: they are the
        # denominators. They must be recognised first, or they would end up
        # overwriting the T=1 column, which has the same ranks and threads.
        if ($2 ~ / seriale$/) { ser[key] = $17; next }
        if ($2 ~ / T\(1\)$/) { t1[key] = $17; next }
        wall[key "," $9] = $17
    }
    END {
        n = split(tlist, ts, " ")
        printf "  %-10s %-5s %-6s %9s %9s", "backend", "simd", "grid",
               "serial", "T(1)"
        for (i = 1; i <= n; i++) printf " %8s", "T=" ts[i]
        printf "\n"
        for (k = 1; k <= nk; k++) {
            split(keys[k], p, ",")
            printf "  %-10s %-5s %-6s %9s %9s", p[1], p[2], p[3],
                   (keys[k] in ser ? sprintf("%.0f", ser[keys[k]]) : "-"),
                   (keys[k] in t1  ? sprintf("%.0f", t1[keys[k]])  : "-")
            for (i = 1; i <= n; i++) {
                kk = keys[k] "," ts[i]
                if (kk in wall) printf " %8.1f", wall[kk]
                else printf " %8s", "-"
            }
            printf "\n"
        }
        print ""
        print "  The numbers are ms per time step, best of the repeats."
        print ""
        print "  serial: MPI not linked, OpenMP not compiled. It is the"
        print "          denominator of the absolute speedup."
        print "  T(1):   the same thing with the parallel machinery inside, on one"
        print "          process and one thread."
        print ""
        print "  The T(1)/serial ratio is the cost of that machinery when idle:"
        print "  as long as it stays within the noise, using T(1) as the"
        print "  denominator is legitimate -- and shown, not asserted."
        print "  Thread speedup must be measured from serial, not from T=1."
    }' "$STUDY_CSV"
fi

study_end
