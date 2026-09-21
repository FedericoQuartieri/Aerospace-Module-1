#!/usr/bin/env bash
#PBS -N nsb-patch-ab
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# How much a change is really worth: the same measurement on two revisions.
#
#   ./scripts/run_patch_ab.sh                 HEAD~1 against HEAD
#   ./scripts/run_patch_ab.sh e64ef95 HEAD    any two revisions
#   qsub scripts/run_patch_ab.sh              on the cluster
#
# ----------------------------------------------------------------------------
# Why measuring afterwards is not enough
# ----------------------------------------------------------------------------
#
# A 5% gain cannot be seen by comparing a number from today with one written in
# a log three weeks ago: in between, the compiler, the node neighbours and the
# frequency of the machine have changed. The only comparison that holds is
# between two binaries built now, from the same sources except for the change,
# launched alternately in the same job.
#
# Hence three choices:
#
#   two real trees      each revision is extracted with `git archive' into a
#                       folder of its own and compiled there. No checkout, no
#                       stash: the working tree is not touched.
#   alternating runs    before-after-before-after, not all the firsts and then
#                       all the seconds. If the machine slows down halfway
#                       through the job, the slowdown is split between the two
#                       instead of being handed to one.
#   every repetition    all the runs end up in the CSV, not the best one. A gain
#                       smaller than the dispersion is not a gain, and to say
#                       so one must see the dispersion.
#
# ----------------------------------------------------------------------------
# The per-stage columns
# ----------------------------------------------------------------------------
#
# print_stats times separately the three momentum steps, the three pressure
# ones, the update, the filling of the permeability and what is attributed to
# nobody. The CSV keeps them all, and that is the point: a change that touches
# a single stage must show up in that stage and in no other. The other stages
# are the control group that the comparison carries along for free.

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}" || exit 1

PRIMA="${1:-HEAD~1}"
DOPO="${2:-HEAD}"

base="build/patch-ab"
csv="$base/results.csv"
log="$base/run.log"
mkdir -p "$base"
exec > >(tee -a "$log") 2>&1

GRIDS="${GRIDS:-128 224}"
REPEATS="${REPEATS:-3}"
# The configurations: rank x thread. Both those where the change acts and those
# where it must not act are needed.
CONFIGS="${CONFIGS:-1x1 1x14 1x56 8x7 56x1}"
BACKENDS="${BACKENDS:-schur pipeline}"
SIMDS="${SIMDS:-0 1}"
# The scenarios. It is not a detail: the forcing term is supplied by the
# scenario, and `paper_data' is the only one that publishes its line version
# (forcing_line_fn in src/data.c). On it, a change to the right-hand side pays
# off at most; on `zero_pressure', which does not have that version, one sees
# how much is left without it. Measuring a single scenario means measuring the
# best case and calling it the case.
SCENARI="${SCENARI:-paper_data zero_pressure}"

# The time steps per size: the cost of a run is steps times cells, and the runs
# must all last roughly the same. At 224^3 with a single thread one step
# already costs a few seconds, and ten steps for three repetitions for two
# revisions would be half an hour for a single case.
passi_per()
{
    [[ -n "${STEPS:-}" ]] && { echo "$STEPS"; return; }
    if   [[ "$1" -le  64 ]]; then echo 20
    elif [[ "$1" -le 128 ]]; then echo 10
    elif [[ "$1" -le 192 ]]; then echo 6
    else                          echo 4
    fi
}

sha_prima="$(git rev-parse --short "$PRIMA")"
sha_dopo="$(git rev-parse --short "$DOPO")"

echo "==============================================================="
echo "comparing  $sha_prima ($PRIMA)  against  $sha_dopo ($DOPO)"
echo "==============================================================="
git --no-pager log --oneline "$PRIMA..$DOPO" | sed 's/^/  /'
echo

if [[ "$sha_prima" == "$sha_dopo" ]]; then
    echo "the two revisions are the same: there is nothing to compare" >&2
    exit 1
fi

# ------------------------------------------------------------ the two trees
#
# `git archive' instead of a checkout: the working tree stays where it is, and
# uncommitted changes do not enter the measurement by mistake.

for rev in "$sha_prima" "$sha_dopo"; do
    albero="$base/$rev"
    if [[ ! -d "$albero" ]]; then
        mkdir -p "$albero"
        git archive "$rev" | tar -x -C "$albero"
        echo "extracted $rev into $albero"
    fi
done
echo

# ----------------------------------------------------------------- compile

costruisci()
{
    local rev="$1" backend="$2" simd="$3" omp="$4" mpi="$5"
    local albero="$base/$rev"
    local nome="bench-$backend-s$simd-o$omp-m$mpi"
    local out="$albero/$nome"

    [[ -x "$out" ]] && { printf '%s' "$out"; return 0; }
    if ! make -s -B -C "$albero" TRIDIAG="$backend" SIMD="$simd" OMP="$omp" \
            MPI="$mpi" build/tests/bench > "$albero/$nome.log" 2>&1; then
        echo "build failed: $rev $nome" >&2
        sed 's/^/    /' "$albero/$nome.log" | head -15 >&2
        return 1
    fi
    mv "$albero/build/tests/bench" "$out"
    printf '%s' "$out"
}

# -------------------------------------------------------------------- one run

leggi()
{
    awk '
        /^  eta system/     { eta  = $3 }
        /^    g term/       { g    = $3 }
        /^  zeta system/    { zeta = $3 }
        /^  u system/       { u    = $3 }
        /^  psi system/     { psi  = $3 }
        /^  phi low/        { lo   = $3 }
        /^  phi high/       { hi   = $3 }
        /^  pressure:/      { pr   = $2 }
        /^  porosity:/      { po   = $2 }
        /^  wall per step/  { wall = $4 }
        /^  mpi per step/   { mpi  = $4 }
        /^  per cell-step/  { cell = $3 }
        /^  bench peak rss/ { rss  = $4 }
        END {
            if (wall == "") { exit 1 }
            sum = eta + zeta + u + psi + lo + hi + pr + po
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.3f,%s,%s,%s",
                   wall, mpi, eta, zeta, u, psi, lo, hi, pr, po,
                   wall - sum, cell, rss, g
        }'
}

corri()
{
    local exe="$1" config="$2" ranks="$3" threads="$4" mpi="$5" scenario="$6"
    local -a comando

    if [[ "$mpi" == "0" ]]; then
        comando=("$exe" "$config")
    else
        comando=("${MPIRUN:-mpirun}" --bind-to none
                 -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES
                 -x BENCH_SCENARIO
                 -n "$ranks" "$exe" "$config")
    fi
    BENCH_SCENARIO="$scenario" \
    OMP_NUM_THREADS="$threads" OMP_PLACES=cores \
        OMP_PROC_BIND="$( [[ "$threads" -gt 1 ]] && echo close || echo false )" \
        timeout 900 "${comando[@]}" 2>&1
}

# The CSV is written at the end and not from scratch: the real measurement
# lasts longer than the walltime of a job, and resubmitting must continue
# instead of starting over. A case that is already complete -- all the
# repetitions, for both revisions -- is skipped.
if [[ ! -f "$csv" || "${FRESH:-0}" == "1" ]]; then
    printf 'revision,sha,scenario,backend,simd,omp,mpi,ranks,threads,nx,steps,repetition,'\
'wall_ms,mpi_ms,eta_ms,zeta_ms,u_ms,psi_ms,philow_ms,phihigh_ms,pressure_ms,'\
'porosity_ms,untimed_ms,cellstep_1e8s,rss_mb,g_ms\n' > "$csv"
fi

gia_fatto()
{
    local scenario="$1" backend="$2" simd="$3" ranks="$4" threads="$5" n="$6"
    local quante atteso=$(( REPEATS * 2 ))

    quante="$(awk -F, -v sc="$scenario" -v b="$backend" -v s="$simd" \
                   -v r="$ranks" -v t="$threads" -v n="$n" '
        NR > 1 && $3 == sc && $4 == b && $5 == s && $8 == r && $9 == t &&
        $10 == n { c++ }
        END { print c + 0 }' "$csv")"
    [[ "$quante" -ge "$atteso" ]]
}

for n in $GRIDS; do
    passi="$(passi_per "$n")"
    config="$base/grid-$n.txt"
    printf 'width = %s\nheight = %s\ndepth = %s\nsteps = %s\nt_end = 1e-1\n' \
        "$n" "$n" "$n" "$passi" > "$config"

    for scenario in $SCENARI; do
      for backend in $BACKENDS; do
        for simd in $SIMDS; do
            for spec in $CONFIGS; do
                ranks="${spec%x*}"
                threads="${spec#*x}"
                omp=1
                mpi=1
                printf '  %-16s %-9s %-6s simd=%s %s^3  ' \
                    "$scenario" "$backend" "$spec" "$simd" "$n"

                if gia_fatto "$scenario" "$backend" "$simd" "$ranks" \
                             "$threads" "$n"; then
                    echo "already done"
                    continue
                fi

                declare -A exe=()
                saltare=0
                for rev in "$sha_prima" "$sha_dopo"; do
                    if ! exe[$rev]="$(costruisci "$rev" "$backend" "$simd" \
                                                 "$omp" "$mpi")"; then
                        saltare=1
                    fi
                done
                [[ "$saltare" == "1" ]] && { echo "skipped"; continue; }

                # Alternate: before, after, before, after...
                for (( r = 1; r <= REPEATS; r++ )); do
                    for rev in "$sha_prima" "$sha_dopo"; do
                        out="$(corri "${exe[$rev]}" "$config" "$ranks" \
                                     "$threads" "$mpi" "$scenario")" || true
                        riga="$(leggi <<< "$out")" || riga=""
                        [[ -z "$riga" ]] && continue
                        etichetta="$( [[ "$rev" == "$sha_prima" ]] \
                                      && echo before || echo after )"
                        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                            "$etichetta" "$rev" "$scenario" "$backend" \
                            "$simd" "$omp" "$mpi" "$ranks" "$threads" "$n" \
                            "$passi" "$r" "$riga" >> "$csv"
                    done
                done
                echo "done"
            done
        done
      done
    done
done

echo
echo "=== result: median of $REPEATS runs, by stage ==="
awk -F, '
function mediana(chiave, quante,   i, v, n) {
    n = conta[chiave]
    if (n == 0) return ""
    # The values arrive in run order; the median wants them sorted.
    for (i = 1; i <= n; i++) v[i] = val[chiave "," i]
    for (i = 2; i <= n; i++) {
        t = v[i]; j = i - 1
        while (j > 0 && v[j] > t) { v[j + 1] = v[j]; j-- }
        v[j + 1] = t
    }
    return (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2
}
NR == 1 {
    for (i = 1; i <= NF; i++) col[$i] = i
    next
}
{
    caso = $col["scenario"] "|" $col["backend"] "|" $col["simd"] "|" $col["ranks"] "x" $col["threads"] "|" $col["nx"]
    if (!(caso in visto)) { casi[++nc] = caso; visto[caso] = 1 }
    for (s = 1; s <= ns; s++) {
        nome = stadi[s]
        k = caso "|" $1 "|" nome
        conta[k]++
        val[k "," conta[k]] = $col[nome] + 0
    }
}
BEGIN { ns = split("wall_ms eta_ms g_ms zeta_ms u_ms", stadi, " ") }
END {
    printf "  %-16s %-9s %-5s %-8s %-6s", "scenario", "backend", "simd", "rankxthr", "n"
    for (s = 1; s <= ns; s++) printf " %19s", stadi[s]
    printf "\n"
    for (c = 1; c <= nc; c++) {
        split(casi[c], p, "|")
        printf "  %-16s %-9s %-5s %-8s %-6s", p[1], p[2], p[3], p[4], p[5]
        for (s = 1; s <= ns; s++) {
            a = mediana(casi[c] "|before|" stadi[s])
            b = mediana(casi[c] "|after|" stadi[s])
            if (a == "" || b == "" || a == 0) { printf " %19s", "-"; continue }
            printf " %8.1f>%-6.1f %3.0f%%", a, b, 100 * (a - b) / a
        }
        printf "\n"
    }
    print ""
    print "  before>after and the percentage saved. The zeta and u stages are"
    print "  the control: a change that touches only eta must leave them still,"
    print "  and how much they move is the measure of this node\x27s noise."
    print ""
    print "  Rows of different scenarios are not comparable in time -- they are"
    print "  different problems -- but their PERCENTAGES are: that difference"
    print "  says how much of the change depends on the scenario."
}' "$csv"

echo
echo
echo "If the walltime killed the job before the end, resubmitting continues"
echo "from where it got to: complete cases are skipped."
echo "FRESH=1 starts over."
echo
echo "csv: $csv"
echo "log: $log"
echo "plots:  ./scripts/plot_patch_ab.py $csv"
