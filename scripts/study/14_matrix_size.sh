#!/usr/bin/env bash
#PBS -N nsb-14-size
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Phase 14 -- the problem size: strong scaling, weak scaling, and the wall.
#
# Three questions that want the same measurements and so it is convenient to do
# them together.
#
#   strong    at fixed problem, how much the time shortens by adding units.
#             It is the question everybody asks and the least honest on a
#             stencil: the local block shrinks until it fits in the cache and
#             the speedup becomes superlinear for a reason that is not
#             parallelism.
#
#   weak      at fixed work per unit, the time stays constant while the
#             problem grows. On a bandwidth-limited code it is the measurement
#             that really says whether the parallelism works.
#
#   the wall  the cost per cell as a function of the size, at fixed
#             parallelism. It grows when the local block leaves the cache, and
#             the point where it grows is a property of the machine, not of
#             the code: knowing it is what allows reading the other two.
#
# The peak memory travels in the CSV (column rss_mb), and for the pipeline it
# is not a detail: it keeps c' and d' of the whole local block for three
# components, so it pays in memory what Schur pays in arithmetic. This phase is
# the only place where that price is seen along an axis.
#
#   qsub scripts/study/14_matrix_size.sh

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub must be run from the repo root; I am in $PWD" >&2
    exit 1
fi

source scripts/study/lib.sh
source scripts/study/matrix.sh

STUDY_CHAIN_MAX="$MATRIX_CHAIN_MAX"
study_begin 14_matrix_size
study_machine

SIZES="${SIZES:-32 48 64 96 128 160 192 224 256}"
# The placements over which the ladder of sizes is walked: serial, the whole
# machine with threads, the whole machine with ranks, and the mixed one.
SIZE_PLACEMENTS="${SIZE_PLACEMENTS:-1x1 1x56 56x1 8x7}"
# Strong scaling: the size stays, the units grow.
STRONG_GRID="${STRONG_GRID:-224}"
STRONG_RANKS="${STRONG_RANKS:-1 2 4 7 8 14 28 56}"
# Weak scaling: constant cells per rank. Every entry is "rank : grid".
WEAK_CONFIGS="${WEAK_CONFIGS:-1:64 64 64|2:128 64 64|4:128 128 64|8:128 128 128|28:224 224 128|56:224 224 224}"

CASE_OMP=1
CASE_MPI=1
CASE_REPEATS="${REPEATS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-1500}"

# The references of the whole phase, once per (backend, simd, size) and not per
# placement: the placement at one process and one thread does not exist. They
# also cover strong scaling (STRONG_GRID is in SIZES) and the base of weak
# scaling, which starts from one rank.
echo "=== the baselines: serial and single process, for every size ==="
for backend in $MATRIX_BACKENDS; do
    for simd in $MATRIX_SIMD; do
        for n in $SIZES; do
            study_baseline label="$backend s$simd N=$n" \
                backend="$backend" simd="$simd" grid="$n $n $n" \
                steps="$(matrix_steps "$n")"
        done
    done
done
echo

echo "=== the wall: cost per cell as the size grows ==="
for place in $SIZE_PLACEMENTS; do
    r="${place%x*}"
    t="${place#*x}"
    [[ $(( r * t )) -le "$(matrix_units)" ]] || continue
    shape="$(study_auto_shape "$r")"
    [[ -n "$shape" ]] || continue

    for backend in $MATRIX_BACKENDS; do
        for simd in $MATRIX_SIMD; do
            for n in $SIZES; do
                matrix_shape_fits "$shape" "$n $n $n" || continue
                study_case label="$backend $place s$simd N=$n" \
                    backend="$backend" simd="$simd" ranks="$r" threads="$t" \
                    shape="$shape" grid="$n $n $n" \
                    steps="$(matrix_steps "$n")"
            done
        done
    done
done
echo

echo "=== strong scaling: ${STRONG_GRID}^3, growing units ==="
steps="$(matrix_steps "$STRONG_GRID")"
grid="$STRONG_GRID $STRONG_GRID $STRONG_GRID"
for backend in $MATRIX_BACKENDS; do
    for n in $STRONG_RANKS; do
        shape="$(study_auto_shape "$n")"
        [[ -n "$shape" ]] || continue
        matrix_shape_fits "$shape" "$grid" || continue
        # Two ways of spending the same units: all ranks, or one rank with as
        # many threads. The difference is the cost of dividing.
        study_case label="$backend forte R=$n T=1" backend="$backend" \
            ranks="$n" threads=1 shape="$shape" simd=1 \
            grid="$grid" steps="$steps"
        study_case label="$backend forte R=1 T=$n" backend="$backend" \
            ranks=1 threads="$n" shape="1 1 1" simd=1 \
            grid="$grid" steps="$steps"
    done
done
echo

echo "=== weak scaling: constant cells per rank ==="
IFS='|' read -r -a weak <<< "$WEAK_CONFIGS"
for entry in "${weak[@]}"; do
    n="${entry%%:*}"
    grid="${entry#*:}"
    shape="$(study_auto_shape "$n")"
    [[ -n "$shape" ]] || continue
    matrix_shape_fits "$shape" "$grid" || continue
    read -r wx wy wz <<< "$grid"

    for backend in $MATRIX_BACKENDS; do
        for simd in $MATRIX_SIMD; do
            study_case label="$backend debole R=$n s$simd" \
                backend="$backend" simd="$simd" ranks="$n" threads=1 \
                shape="$shape" grid="$grid" \
                steps="$(matrix_steps "$wx")" \
                note="debole, ${wx}x${wy}x${wz}"
        done
    done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== cost per cell (1e-8 s) and peak memory, by size ==="
    awk -F, -v phase=14_matrix_size '
    NR == 1 || $1 != phase || $(NF - 1) != "ok" || $2 !~ / N=/ { next }
    # The reference labels also contain " N=".
    $2 ~ / (seriale|T\(1\))$/ { next }
    {
        k = $3 "," $5 "," $8 "x" $9
        cell[k "," $10] = $28 + 0
        rss[k "," $10] = $29 + 0
        if (!(k in seen)) { keys[++nk] = k; seen[k] = 1 }
        if (!($10 in seen_n)) { ns[++nn] = $10 + 0; seen_n[$10] = 1 }
    }
    END {
        printf "  %-10s %-5s %-8s", "backend", "simd", "rankxthr"
        for (j = 1; j <= nn; j++) printf " %8s", ns[j]
        printf "\n"
        for (i = 1; i <= nk; i++) {
            split(keys[i], p, ",")
            printf "  %-10s %-5s %-8s", p[1], p[2], p[3]
            for (j = 1; j <= nn; j++) {
                kk = keys[i] "," ns[j]
                if (kk in cell) printf " %8.2f", cell[kk]
                else printf " %8s", "."
            }
            printf "\n"
        }
        print ""
        print "  Cost per cell per step: if it were only arithmetic it would stay"
        print "  constant. Where it rises, the local block has left the cache."
    }' "$STUDY_CSV"
fi

study_end
