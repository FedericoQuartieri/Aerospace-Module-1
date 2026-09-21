#!/usr/bin/env bash
#PBS -N nsb-core0
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Is the serial binary slower on one core than on another?
#
# In the second campaign, on cpu04 and only there, the serial came out slower
# than T(1), up to 15% at 224^3; on the other nodes T(1) costs 2-3% more, as it
# should. The study pins the serial with `taskset -c 0' and does not bind T(1)
# to anything, and core 0 is usually the one that serves the interrupts.
#
# This job measures the same binary on different cores of the same node, and
# counts the interrupts of each core while measuring. It must be launched on
# the suspect node and on a control one:
#
#   qsub -l select=1:ncpus=112:host=cpu04 scripts/study/check_core0.sh
#   qsub -l select=1:ncpus=112:host=cpu03 scripts/study/check_core0.sh
#
# The result ends up in build/study/check_core0/<node>.txt.

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1
if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub must be run from the repository root; this is $PWD" >&2
    exit 1
fi
source scripts/study/lib.sh
set +e
export LC_ALL=C

GRID="${GRID:-224}"
STEPS="${STEPS:-4}"
REPEATS="${REPEATS:-3}"
MPIRUN_="${MPIRUN:-mpirun}"

STUDY_OUT="$STUDY_BASE/check_core0"
mkdir -p "$STUDY_OUT" "$STUDY_BIN"
node="$(hostname -s)"
exec > >(tee "$STUDY_OUT/$node.txt") 2>&1

# Three CPUs: core 0, another physical core of the same socket (not its SMT
# twin), the first core of the other socket. lscpu -p numbers by logical CPU.
read -r c0 c_stesso c_altro < <(lscpu -p=CPU,CORE,SOCKET | awk -F, '
    /^#/ { next }
    { n++; cpu[n] = $1; core[n] = $2; sock[n] = $3 }
    $1 == 0 { core0 = $2; sock0 = $3 }
    END {
        for (i = 1; i <= n; i++) if (sock[i] == sock0 && core[i] != core0) { stesso = cpu[i]; break }
        for (i = 1; i <= n; i++) if (sock[i] != sock0) { altro = cpu[i]; break }
        print 0, stesso, altro
    }')

# The interrupts served by each logical CPU, summed over all the rows.
interrupts()
{
    awk 'NR == 1 { n = NF; next }
         { for (i = 2; i <= n + 1; i++) if ($i ~ /^[0-9]+$/) s[i - 2] += $i }
         END { for (c = 0; c < n; c++) print c, s[c] + 0 }' /proc/interrupts
}

# The best time over REPEATS runs, in ms per step, in MIS; empty if the program
# did not print the times.
misura()
{
    local out wall r
    MIS=""
    for (( r = 0; r < REPEATS; r++ )); do
        out="$(OMP_NUM_THREADS=1 "$@" "$config" 2>&1 < /dev/null)"
        wall="$(study_parse <<< "$out" | cut -d, -f4)"
        if [[ -z "$wall" ]]; then
            sed 's/^/      /' <<< "$out" | tail -5
            MIS=""
            return
        fi
        if [[ -z "$MIS" ]] || awk "BEGIN { exit !($wall < $MIS) }"; then
            MIS="$wall"
        fi
    done
}

riga()
{
    if [[ -z "$3" ]]; then
        printf '  %-34s %5s %12s\n' "$1" "$2" "no timings"
        return
    fi
    printf '  %-34s %5s %12.1f %8.3f\n' "$1" "$2" "$3" \
        "$(awk "BEGIN { print $3 / $4 }")"
}

echo "=== core-0 check on $node -- $(date '+%Y-%m-%d %H:%M:%S') ==="
lscpu | grep -E '^(Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|NUMA node[0-9]+ CPU)'
printf 'cpus used:  core 0 = cpu%s, same socket = cpu%s, other socket = cpu%s\n' \
    "$c0" "${c_stesso:--}" "${c_altro:--}"
for c in $c0 $c_stesso $c_altro; do
    printf '  cpu%-4s governor %-12s frequency %s kHz\n' "$c" \
        "$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null || echo -)" \
        "$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null || echo -)"
done

config="$(STUDY_OUT="$STUDY_OUT" study_config "$GRID" "$GRID" "$GRID" "$STEPS")"
prima="$(interrupts)"
for simd in 1 0; do
    ser="$(study_build schur "$simd" 0 0 64)" || exit 1
    par="$(study_build schur "$simd" 0 1 64)" || exit 1
    echo
    echo "=== schur simd=$simd, ${GRID}^3, $STEPS steps, best of $REPEATS ==="
    printf '  %-34s %5s %12s %8s\n' variant cpu "ms/step" "/unpinned"
    misura "$ser"; libero="$MIS"
    if [[ -z "$libero" ]]; then
        echo "  the unpinned serial printed no timings"
        continue
    fi
    riga "serial, unpinned" "-" "$libero" "$libero"
    misura taskset -c "$c0" "$ser";         riga "serial, core 0 (as the study)" "$c0" "$MIS" "$libero"
    if [[ -n "$c_stesso" ]]; then
        misura taskset -c "$c_stesso" "$ser"; riga "serial, other core, same socket" "$c_stesso" "$MIS" "$libero"
    fi
    if [[ -n "$c_altro" ]]; then
        misura taskset -c "$c_altro" "$ser";  riga "serial, other socket" "$c_altro" "$MIS" "$libero"
    fi
    misura "$MPIRUN_" --bind-to none -n 1 "$par"
    riga "T(1), unbound (as the study)" "-" "$MIS" "$libero"
    misura "$MPIRUN_" --bind-to none -n 1 taskset -c "$c0" "$par"
    riga "T(1), core 0" "$c0" "$MIS" "$libero"
done
dopo="$(interrupts)"

echo
echo "=== interrupts served while measuring, busiest cpus ==="
join <(sort -k1,1 <<< "$prima") <(sort -k1,1 <<< "$dopo") \
    | awk '{ print $1, $3 - $2 }' | sort -k2,2nr \
    | awk -v c0="$c0" '{ if ($1 == c0) rank = NR } NR <= 6 { printf "  cpu%-4s %10d\n", $1, $2 }
                       END { printf "  cpu%s is number %d of %d\n", c0, rank, NR }'
