#!/usr/bin/env bash
#PBS -N nsb-11-mpi
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Phase 11 -- one thread per rank, and EVERY shape of the process grid.
#
# Phase 03 measures increasing ranks with a single shape per number, chosen by
# hand; phase 04 compares a few shapes. This one takes them all: for each
# number of ranks, all the triples (px, py, pz) whose product is that number.
# There are 80 shapes in all between 1 and 56 ranks.
#
# The permutations are not duplicates, and that is the point of the phase. The
# three axes are not interchangeable:
#
#   x  is the direction contiguous in memory. Schur, when it divides it, loses
#      the whole-line kernels along the other two? no: it loses those along x,
#      which it does not have, and pays three local solves instead of one. The
#      pipeline puts the NON-adjacent lines there, so along x it never
#      vectorises.
#   y, z  are the directions where the vector kernels work, and dividing them
#      means losing them -- for Schur. Not for the pipeline: its kernel
#      works across the lines and survives the division.
#
# So the same number of processes, arranged differently, should give different
# times, and different in the OPPOSITE way for the two backends. If it does not
# happen, the effect is below the noise of the memory bandwidth and this must
# be said.
#
# ----------------------------------------------------------------------------
# Why the cubic grid alone is not enough
# ----------------------------------------------------------------------------
#
# On a cubic global grid, the shape (1, 1, 56) does not give just "z divided":
# it also gives a local block of 224 x 224 x 4, that is a slab. The time that
# comes out mixes two things -- which axis is cut and what shape the block has
# taken -- and from the number they cannot be separated.
#
# For this reason the phase measures the same question in three ways, and they
# are three different blocks below:
#
#   1. cubic global grid      as in production. It is the real case, and the
#                             two effects stay together.
#   2. cubic local block      the global grid follows the shape: with B = 96
#                             and shape (p, q, r) the global one is 96p x 96q x
#                             96r. Every shape then has the SAME local block
#                             and the SAME work per process, and the only thing
#                             that changes is which axis carries the cuts. It
#                             is here that the effect of the axis can be read
#                             on its own.
#   3. shape of the block     fixed process shape, stretched global grid: same
#                             number of cells per process, long or flat local
#                             block. It is the other half of what case 1 kept
#                             together.
#
#   qsub scripts/study/11_matrix_mpi.sh
#   RANKS="8 56" qsub scripts/study/11_matrix_mpi.sh

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub must be run from the repo root; I am in $PWD" >&2
    exit 1
fi

source scripts/study/lib.sh
source scripts/study/matrix.sh

STUDY_CHAIN_MAX="$MATRIX_CHAIN_MAX"
study_begin 11_matrix_mpi
study_machine

# The size where everything is tried, shapes for both SIMD settings.
FULL_GRID="${FULL_GRID:-224}"
# The sizes where all the shapes are tried but only with SIMD on: they serve to
# see whether the ranking of the shapes depends on the size.
PLAIN_GRIDS="${PLAIN_GRIDS:-128}"
RANKS="${RANKS:-$MATRIX_RANKS}"
# Block 2: the side of the local block, which stays the same for every shape.
BLOCK="${BLOCK:-96}"
# Block 3: local blocks of equal volume and different shape, at fixed process
# shape. 2 097 152 cells per process in all the rows.
ASPETTI="${ASPETTI:-128 128 128|256 128 64|64 128 256|512 64 64|64 512 64|64 64 512}"
ASPETTO_RANKS="${ASPETTO_RANKS:-8}"

CASE_THREADS=1
# Without OpenMP: here the threads do not matter, and the threaded build costs
# about 1% even with a single thread. The bridge rows at the bottom stitch it
# back to phase 12.
CASE_OMP=0
CASE_MPI=1
CASE_REPEATS="${REPEATS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-900}"

emit_shapes()
{
    local n="$1" grid="$2" simd="$3" steps="$4" backend shape
    # All the shapes before the loop: study_case launches the solver, which
    # reads stdin and would leave the loop without the list after the first
    # case.
    local forme=()
    mapfile -t forme < <(matrix_shapes "$n")

    for backend in $MATRIX_BACKENDS; do
        for shape in "${forme[@]}"; do
            matrix_shape_fits "$shape" "$grid" || continue
            study_case label="$backend R=$n ${shape// /x} s$simd" \
                backend="$backend" ranks="$n" shape="$shape" simd="$simd" \
                grid="$grid" steps="$steps"
        done
    done
}

steps="$(matrix_steps "$FULL_GRID")"
grid="$FULL_GRID $FULL_GRID $FULL_GRID"
for simd in $MATRIX_SIMD; do
    echo "=== ${FULL_GRID}^3, simd=$simd: every shape of every rank count ==="
    for backend in $MATRIX_BACKENDS; do
        study_baseline label="$backend ${FULL_GRID}^3 s$simd" \
            backend="$backend" simd="$simd" grid="$grid" steps="$steps"
    done
    for n in $RANKS; do
        emit_shapes "$n" "$grid" "$simd" "$steps"
    done
    echo
done

for m in $PLAIN_GRIDS; do
    steps="$(matrix_steps "$m")"
    grid="$m $m $m"
    echo "=== ${m}^3, simd=1: the same shapes on a different size ==="
    for backend in $MATRIX_BACKENDS; do
        study_baseline label="$backend ${m}^3 s1" \
            backend="$backend" simd=1 grid="$grid" steps="$steps"
    done
    for n in $RANKS; do
        emit_shapes "$n" "$grid" 1 "$steps"
    done
    echo
done

# ---------------------------------------------------------------- block 2
#
# The global grid follows the shape, so every row has the same local block and
# the same work per process. If the axis that is cut did not matter, these rows
# would all give the same time: it is a comparison at constant work, not at
# constant problem, and it is the only way to read the axis on its own.

echo "=== local block ${BLOCK}^3 fixed: only the split axis changes ==="
steps="$(matrix_steps "$BLOCK")"
# Here the right reference is the LOCAL BLOCK, not the global grid. These rows
# keep the work per process constant, so the question is "how much does a
# process pay for its block, compared with doing it alone": the denominator is
# a serial run on ${BLOCK}^3.  A serial run on the global grid -- which reaches
# 96x96x5376 -- would answer a question nobody asks, and would cost 6-8 GB and
# minutes at random.
for backend in $MATRIX_BACKENDS; do
    study_baseline label="cubo $backend blocco ${BLOCK}^3" \
        backend="$backend" simd=1 grid="$BLOCK $BLOCK $BLOCK" \
        steps="$steps" note="riferimento a blocco locale"
done
for backend in $MATRIX_BACKENDS; do
    for n in $RANKS; do
        forme=()
        mapfile -t forme < <(matrix_shapes "$n")
        for forma in "${forme[@]}"; do
            read -r px py pz <<< "$forma"
            study_case label="cubo $backend R=$n ${px}x${py}x${pz}" \
                backend="$backend" ranks="$n" shape="$px $py $pz" simd=1 \
                grid="$(( px * BLOCK )) $(( py * BLOCK )) $(( pz * BLOCK ))" \
                steps="$steps" \
                note="blocco locale ${BLOCK}^3, globale al seguito"
        done
    done
done
echo

# ---------------------------------------------------------------- block 3
#
# The other half: fixed process shape, local block of equal volume and
# different proportions. Here the axis that is cut never changes, only whether
# the block is a cube, a bar or a slab changes -- and on a stencil that shape
# decides how many cache lines are reused.

echo "=== same volume per process, local block of a different shape ==="
IFS='|' read -r -a aspetti <<< "$ASPETTI"
for n in $ASPETTO_RANKS; do
    shape="$(study_auto_shape "$n")"
    [[ -n "$shape" ]] || continue
    read -r px py pz <<< "$shape"
    for entry in "${aspetti[@]}"; do
        read -r bx by bz <<< "$entry"
        for backend in $MATRIX_BACKENDS; do
            for simd in $MATRIX_SIMD; do
                # Same reasoning as block 2: the denominator is that same block
                # shape done by a single process.
                study_baseline label="aspetto $backend ${bx}x${by}x${bz} s$simd" \
                    backend="$backend" simd="$simd" \
                    grid="$bx $by $bz" steps="$(matrix_steps "$bx")" \
                    note="riferimento a blocco locale ${bx}x${by}x${bz}"
                study_case label="aspetto $backend ${bx}x${by}x${bz} s$simd" \
                    backend="$backend" ranks="$n" shape="$shape" simd="$simd" \
                    grid="$(( px * bx )) $(( py * by )) $(( pz * bz ))" \
                    steps="$(matrix_steps "$bx")" \
                    note="blocco locale ${bx}x${by}x${bz}"
            done
        done
    done
done
echo

# Bridge rows to phase 12: the same thing, build with OpenMP and a single
# thread.
echo "=== bridge rows: OpenMP build on one thread ==="
for backend in $MATRIX_BACKENDS; do
    for n in 1 8 56; do
        shape="$(study_auto_shape "$n")"
        [[ -n "$shape" ]] || continue
        study_case label="$backend ponte R=$n omp=1" backend="$backend" \
            ranks="$n" shape="$shape" simd=1 omp=1 \
            grid="$FULL_GRID $FULL_GRID $FULL_GRID" \
            steps="$(matrix_steps "$FULL_GRID")"
    done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== the best and the worst shape, for each rank count ==="
    awk -F, -v phase=11_matrix_mpi '
    NR == 1 || $1 != phase || $(NF - 1) != "ok" { next }
    $2 ~ /ponte|^cubo |^aspetto / { next }
    # The references are not shapes: they must be excluded, or they would
    # appear as the best shape of every row, since they run on a single
    # process.
    $2 ~ / (seriale|T\(1\))$/ { next }
    {
        k = $3 "," $5 "," $10 "," $8
        s = $14 "x" $15 "x" $16
        if (!(k in best) || $17 + 0 < best[k]) { best[k] = $17 + 0; bs[k] = s }
        if (!(k in worst) || $17 + 0 > worst[k]) { worst[k] = $17 + 0; ws[k] = s }
        if (!(k in seen)) { keys[++nk] = k; seen[k] = 1 }
    }
    END {
        printf "  %-10s %-5s %-6s %-5s %9s %-9s %9s %-9s %7s\n",
               "backend", "simd", "grid", "rank",
               "best", "shape", "worst", "shape", "spread"
        for (i = 1; i <= nk; i++) {
            split(keys[i], p, ",")
            printf "  %-10s %-5s %-6s %-5s %9.1f %-9s %9.1f %-9s %6.2fx\n",
                   p[1], p[2], p[3], p[4],
                   best[keys[i]], bs[keys[i]],
                   worst[keys[i]], ws[keys[i]],
                   (best[keys[i]] > 0 ? worst[keys[i]] / best[keys[i]] : 0)
        }
        print ""
        print "  The spread is what laying out the same processes badly costs."
        print "  If the best shape is the same for both backends, then what"
        print "  decides is memory bandwidth and not the algorithm."
        print ""
        print "  WARNING: the global grid is cubic here, so a very skewed shape"
        print "  also gives a slab-shaped local block. The two causes are"
        print "  separated in the table below."
    }' "$STUDY_CSV"

    echo
    echo "=== fixed local block: the time of the three pure splits, by axis ==="
    awk -F, -v phase=11_matrix_mpi '
    NR == 1 || $1 != phase || $(NF - 1) != "ok" || $2 !~ /^cubo / { next }
    {
        # Pure cut: all the processes on a single axis.
        asse = ""
        if ($15 == 1 && $16 == 1) asse = "x"
        else if ($14 == 1 && $16 == 1) asse = "y"
        else if ($14 == 1 && $15 == 1) asse = "z"
        if (asse == "" || $8 + 0 < 2) next
        wall[$3 "," $8 "," asse] = $17 + 0
        if (!($3 "," $8 in seen)) { keys[++nk] = $3 "," $8; seen[$3 "," $8] = 1 }
    }
    END {
        printf "  %-10s %-6s %9s %9s %9s %9s\n",
               "backend", "rank", "x", "y", "z", "z/x"
        for (i = 1; i <= nk; i++) {
            split(keys[i], p, ",")
            x = wall[keys[i] ",x"]; y = wall[keys[i] ",y"]; z = wall[keys[i] ",z"]
            printf "  %-10s %-6s", p[1], p[2]
            printf " %9s", (x ? sprintf("%.1f", x) : "-")
            printf " %9s", (y ? sprintf("%.1f", y) : "-")
            printf " %9s", (z ? sprintf("%.1f", z) : "-")
            printf " %9s\n", (x && z ? sprintf("%.2fx", z / x) : "-")
        }
        print ""
        print "  Same local block in every column, same work per process: if"
        print "  the axis did not matter, the three columns would be equal."
        print "  Schur and pipeline should lean in opposite directions."
    }' "$STUDY_CSV"
fi

study_end
