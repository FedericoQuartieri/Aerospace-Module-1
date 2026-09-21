#!/usr/bin/env bash
#PBS -N nsb-12-hybrid
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Phase 12 -- the full rank x thread product.
#
# Phase 05 measures the diagonal: rank x thread constant, equal to the physical
# cores. That diagonal confounds three things -- how many threads each rank
# has, how loaded the machine is, and how the domain is divided -- and indeed
# the reading of MULTITHREAD.md 9.2 broke on it (see hybrid-collapse).
#
# This phase fills the rectangle instead of the diagonal: EVERY (rank, thread)
# pair that fits in the node, for both backends, with and without SIMD. With
# the full rectangle the three things separate by themselves:
#
#   - at fixed ranks, reading along the threads, only the threads change;
#   - at fixed threads, reading along the ranks, only the division changes;
#   - the anti-diagonals at constant product are the rows of phase 05, and now
#     they have around them the context that was missing.
#
# The prediction, from the code: on the pipeline the threads pay off little
# anyway, because every level is a barrier (see PIPELINE, batch and barriers),
# and therefore the many-thread column should be flat and improve with large
# batches -- which is what phase 13 measures. On Schur instead the threads pay
# off as long as the axis is not divided.
#
#   qsub scripts/study/12_matrix_hybrid.sh

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub must be run from the repo root; I am in $PWD" >&2
    exit 1
fi

source scripts/study/lib.sh
source scripts/study/matrix.sh

STUDY_CHAIN_MAX="$MATRIX_CHAIN_MAX"
study_begin 12_matrix_hybrid
study_machine

# The main size, where the rectangle is tried with and without SIMD.
FULL_GRID="${FULL_GRID:-224}"
# The others, where it is tried only with SIMD on.
PLAIN_GRIDS="${PLAIN_GRIDS:-128}"

CASE_OMP=1
CASE_MPI=1
CASE_REPEATS="${REPEATS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-1200}"

emit_rectangle()
{
    local grid="$1" simd="$2" steps="$3" backend r t shape coppia
    # The list is read entirely first: inside the loop there is study_case,
    # which launches the solver, and a program that reads stdin would eat the
    # pairs that remain.
    local coppie=()
    mapfile -t coppie < <(matrix_pairs)

    for backend in $MATRIX_BACKENDS; do
        # The corner of the rectangle: one rank, one thread, and even before
        # that the binary in which MPI and OpenMP are not even compiled. All
        # the pairs below are read with respect to these two.
        study_baseline label="$backend s$simd" \
            backend="$backend" simd="$simd" grid="$grid" steps="$steps"

        for coppia in "${coppie[@]}"; do
            read -r r t <<< "$coppia"
            shape="$(study_auto_shape "$r")"
            [[ -n "$shape" ]] || continue
            matrix_shape_fits "$shape" "$grid" || continue
            study_case label="$backend R=$r T=$t s$simd" \
                backend="$backend" ranks="$r" threads="$t" shape="$shape" \
                simd="$simd" grid="$grid" steps="$steps"
        done
    done
}

steps="$(matrix_steps "$FULL_GRID")"
for simd in $MATRIX_SIMD; do
    echo "=== ${FULL_GRID}^3, simd=$simd: every rank x thread pair ==="
    emit_rectangle "$FULL_GRID $FULL_GRID $FULL_GRID" "$simd" "$steps"
    echo
done

for m in $PLAIN_GRIDS; do
    echo "=== ${m}^3, simd=1: the same rectangle on another size ==="
    emit_rectangle "$m $m $m" 1 "$(matrix_steps "$m")"
    echo
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== the rectangle: ms/step, rows are ranks, columns are threads ==="
    awk -F, -v phase=12_matrix_hybrid -v tlist="$MATRIX_THREADS" '
    NR == 1 || $1 != phase || $(NF - 1) != "ok" { next }
    # The references have one rank and one thread: without this they would end
    # up in the cell R=1 T=1 of the rectangle, where a real case sits.
    $2 ~ / (serial|T\(1\))$/ { next }
    {
        k = $3 "," $5 "," $10
        wall[k "," $8 "," $9] = $17 + 0
        if (!(k in seen)) { keys[++nk] = k; seen[k] = 1 }
        if (!($8 in seen_r)) { rs[++nr] = $8 + 0; seen_r[$8] = 1 }
    }
    END {
        nt = split(tlist, ts, " ")
        for (i = 1; i <= nk; i++) {
            split(keys[i], p, ",")
            printf "\n  %s  simd=%s  %s^3\n", p[1], p[2], p[3]
            printf "  %-6s", "rank"
            for (j = 1; j <= nt; j++) printf " %8s", "T=" ts[j]
            printf "\n"
            for (r = 1; r <= nr; r++) {
                printf "  %-6s", rs[r]
                for (j = 1; j <= nt; j++) {
                    kk = keys[i] "," rs[r] "," ts[j]
                    if (kk in wall) printf " %8.1f", wall[kk]
                    else printf " %8s", "."
                }
                printf "\n"
            }
        }
        print ""
        print "  A dot is a pair that does not fit in the node. The anti-diagonals"
        print "  at constant product are the rows of phase 05."
    }' "$STUDY_CSV"
fi

study_end
