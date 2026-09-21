#!/usr/bin/env bash
#PBS -N nsb-convergence
#PBS -q cpu
#PBS -l select=1:ncpus=28
#PBS -l walltime=04:00:00
#PBS -j oe
#
# The grid convergence study that really matters: 256^3.
#
#   qsub scripts/run_convergence_cluster.sh
#
# It is a wrapper around scripts/run_convergence.sh, which already does all the
# work: spatial refinement at fixed dt, temporal refinement at fixed grid, and
# the computation of the observed orders.
#
# ----------------------------------------------------------------------------
# Why on the `cpu' queue and not on `scalability'
# ----------------------------------------------------------------------------
#
# Because this study measures ERRORS, not times. A node shared with the jobs of
# other users slows the run down but does not move the L2 norms by a single
# digit, so the exclusivity of the node -- which on `scalability' costs the
# half-hour limit -- is of no use here. On `cpu' there are 48 hours of
# walltime, and they are needed: the temporal study at 256^3 adds up 300 time
# steps.
#
# The flip side is the ceiling of 28 CPUs per job on that queue, that is 14
# physical cores. They are enough: 14 cores on 256^3 keep the step under a
# second.
#
# ----------------------------------------------------------------------------
# Why EXTRA_CFLAGS
# ----------------------------------------------------------------------------
#
# run_convergence.sh compiles without SIMD and without threads, which is
# perfectly fine for the small grids locally. At 256^3 for 160 steps it would
# not finish within any reasonable walltime, so here the flags are passed
# through the environment variable that it accepts for this purpose.
#
# The threads do not change the result: the lines are independent and every sum
# stays in the order it had, so the norms are identical to those of a single
# thread. Verified at 32/64/128 locally before writing this.

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

build="build/convergence"
log="$build/cluster.log"
mkdir -p "$build"

exec > >(tee "$log") 2>&1

echo "=== machine ==="
printf 'cpus granted:  %s\n' "$(grep Cpus_allowed_list /proc/self/status | cut -f2)"

# One thread per PHYSICAL core granted: SMT does not add arithmetic units on
# this load and locally it proved counterproductive.
allowed_cpus="$(grep -c ^processor /proc/cpuinfo)"
granted="$(grep Cpus_allowed_list /proc/self/status | cut -f2 |
           awk -F, '{n=0; for(i=1;i<=NF;i++){split($i,r,"-");
                     n += (r[2]=="") ? 1 : r[2]-r[1]+1} print n}')"
threads="${OMP_NUM_THREADS:-$(( granted / 2 ))}"
[[ "$threads" -lt 1 ]] && threads=1
printf 'threads used:  %s\n' "$threads"
echo

echo "=== convergence study at 256^3 ==="
echo "  spatial: 32, 64, 128, 256 at fixed dt"
echo "  temporal: 256^3 with dt halved four times (300 steps in total)"
echo

export OMP_NUM_THREADS="$threads"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export EXTRA_CFLAGS="-mavx2 -mfma -DUSE_SIMD -fopenmp -DUSE_OMP"

./scripts/run_convergence.sh

echo
echo "log in $log"
