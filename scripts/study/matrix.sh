# shellcheck shell=bash
#
# The part common to phases 10-15, the exhaustive campaign.
#
# The study that preceded this campaign asked one question per phase and varied
# one thing at a time; it has been removed, and these six contain it. They fill
# the matrix: every backend for every SIMD for every number of ranks for every
# shape of the process grid for every number of threads, and the sizes on top
# of everything. There is no prediction to falsify, there is a surface to
# measure, and its value is that afterwards one can ask questions that today we
# do not yet know we have.
#
# Six phases and not one because each is a PBS job of its own: they all ask for
# a whole node, so PBS puts them on six different nodes and they run together.
# Each resubmits itself until it is done, and each resumes from where it had
# got to, so the campaign lasts as long as it needs without anybody following
# it.
#
#   10_matrix_threads   one rank, increasing threads: the machine on its own
#   11_matrix_mpi       one thread, increasing ranks, EVERY shape of the grid
#   12_matrix_hybrid    the full rank x thread product
#   13_matrix_batch     the pipeline batch against the placement
#   14_matrix_size      the problem size, strong and weak
#   15_matrix_check     the norms: the matrix still solves the right problem
#
# They all write the same columns of lib.sh, so `./scripts/run_study.sh merge'
# merges them into a single CSV. `./scripts/plot_matrix.py' draws from that
# one.
#
# How long it takes: with the defaults it is about 1800 cases. Spread over six
# chains in parallel it is a few hours each. To really use four days the axes
# are widened from the environment, which is why they are all variables:
#
#   GRIDS="128 224 256" REPEATS=3 ./scripts/run_study.sh submit
#
# A warning that holds for the whole campaign: there is no multi-node here. On
# this cluster there is no working way to launch processes on several nodes --
# neither plm tm, nor ssh, nor pbs_tmrsh -- so the ceiling is one node: 56
# physical cores, 112 logical cpus. Every case that asks for more than 112
# units is not emitted.

# The values that come back in several phases. They are divisors of 56 because
# that is the number of physical cores of the node: numbers that do not divide
# it leave a socket unpaired and measure the placement instead of the
# algorithm.
MATRIX_RANKS="${MATRIX_RANKS:-1 2 4 7 8 14 28 56}"
MATRIX_THREADS="${MATRIX_THREADS:-1 2 4 7 8 14 28 56 112}"
MATRIX_BACKENDS="${MATRIX_BACKENDS:-schur pipeline}"
MATRIX_SIMD="${MATRIX_SIMD:-0 1}"

# The ceiling of units (rank x thread). Normally the logical cpus of the node;
# it is lowered by hand to try the campaign on a small machine.
matrix_units()
{
    echo "${MATRIX_UNITS:-${STUDY_LOGICAL:-$(nproc --all)}}"
}

# How many time steps for a size. The product steps x cells is what decides how
# long a case lasts, and the cases must all last roughly the same: otherwise
# the bulk of the campaign goes in two rows.
matrix_steps()
{
    local n="$1"

    if [[ -n "${STEPS:-}" ]]; then
        echo "$STEPS"
        return
    fi
    if   [[ "$n" -le  64 ]]; then echo 20
    elif [[ "$n" -le 128 ]]; then echo 10
    elif [[ "$n" -le 192 ]]; then echo 6
    else                          echo 4
    fi
}

# ALL the shapes (px, py, pz) with px*py*pz = n, in axis order.
#
# The permutations are NOT duplicates. The three axes are not interchangeable:
# x is the direction contiguous in memory, and the two backends treat it
# differently -- Schur loses the whole-line kernels there as soon as it divides
# it, the pipeline puts the non-adjacent lines there. "Which axis is divided"
# is exactly one of the variables that this campaign must measure, not one to
# be factored away.
matrix_shapes()
{
    local n="$1" px py pz rest

    for (( px = 1; px <= n; px++ )); do
        (( n % px == 0 )) || continue
        rest=$(( n / px ))
        for (( py = 1; py <= rest; py++ )); do
            (( rest % py == 0 )) || continue
            pz=$(( rest / py ))
            printf '%d %d %d\n' "$px" "$py" "$pz"
        done
    done
}

# The (rank, thread) pairs that fit in the node.
matrix_pairs()
{
    local units r t
    units="$(matrix_units)"

    for r in $MATRIX_RANKS; do
        for t in $MATRIX_THREADS; do
            (( r * t <= units )) && printf '%d %d\n' "$r" "$t"
        done
    done
}

# A shape must not ask for more blocks than cells along an axis: decomp shuts
# the program down if it does, and it would be a failed case instead of a
# skipped case.
matrix_shape_fits()
{
    local px py pz nx ny nz
    read -r px py pz <<< "$1"
    read -r nx ny nz <<< "$2"

    [[ "$px" -le "$nx" && "$py" -le "$ny" && "$pz" -le "$nz" ]]
}

# The phases of the campaign last longer than a normal chain: four days of
# half-hour jobs are almost two hundred links. The ceiling stays, because it
# serves to stop a chain that runs idle, but it sits much higher.
MATRIX_CHAIN_MAX="${MATRIX_CHAIN_MAX:-200}"
