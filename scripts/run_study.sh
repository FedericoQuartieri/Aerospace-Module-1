#!/usr/bin/env bash
#
# The scaling campaign: six phases, one command.
#
#   ./scripts/run_study.sh submit          submits every phase to PBS
#   ./scripts/run_study.sh submit 11 13    only some of them
#   ./scripts/run_study.sh local 10        runs one phase here and now
#   ./scripts/run_study.sh dry             lists the cases without running them
#   ./scripts/run_study.sh merge           merges the CSVs and draws the plots
#   ./scripts/run_study.sh status          how far along the phases are
#   ./scripts/run_study.sh probe           what each queue grants, measured
#
# The six phases are independent and run in parallel: each one builds the
# binaries it needs and asks for a whole node, so PBS will never put two of
# them on the same machine to disturb each other. Phase 15 is the one that
# checks that every variant gives the same answer, and it is worth reading
# first: if it does not, the timings of the others compare different programs.
#
# The `scalability' queue grants 30 minutes per job (resources_max.walltime =
# 00:30:00) and the study wants far more. Nothing needs to be done: each phase
# works to a budget, stops before being killed and resubmits itself until it
# is done. `status' says how far it got. Submitting a phase that is already
# running does no harm -- the cases already measured are skipped -- so
# `submit' doubles as "continue".
#
# Useful variables, passed like this:
#
#   GRIDS=128 REPEATS=1 ./scripts/run_study.sh submit 11
#   WALLTIME=12:00:00 ./scripts/run_study.sh submit 05
#   STUDY_HOST=cpu03 ./scripts/run_study.sh submit 10 12 14   (every job on cpu03)
#
# Environment variables reach the job through `qsub -V', so it is enough to
# put them in front of the command. STUDY_ENV="key=value ..." does the same
# for whoever prefers it, but the values cannot contain spaces.
#
#   FRESH=1     start over instead of resuming
#   STUDY_BUDGET=1500  seconds of useful work per job: 25 of the 30 minutes
#               granted, the rest is margin to close the case in flight
#   AUTO_RESUBMIT=0    do not resubmit, stop when the budget is spent
#   DRY_RUN=1   list the cases and run nothing
#   REPEATS     repeats per case (default 2, the best one is kept)
#   STEPS       time steps per case
#   CASE_BATCH  pipeline batch of the phases that do not sweep it: auto (the
#               default, chosen at start-up as the solver does) or a number

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
phases_dir="$root/scripts/study"
action="${1:-help}"
shift || true

# The six phases fill the matrix instead of answering one question at a time:
# they last days and resubmit themselves. See scripts/study/matrix.sh. The
# ten-phase targeted study that preceded them has been removed -- it remains in
# the git history -- because what it measured is measured by these, in full.
all_phases=(10_matrix_threads 11_matrix_mpi 12_matrix_hybrid \
            13_matrix_batch 14_matrix_size 15_matrix_check)

# A numeric prefix is enough to choose a phase: `05' stands for `05_hybrid'.
resolve_phases()
{
    local wanted=("$@") phase name found
    if [[ ${#wanted[@]} -eq 0 ]]; then
        printf '%s\n' "${all_phases[@]}"
        return
    fi
    for name in "${wanted[@]}"; do
        found=0
        for phase in "${all_phases[@]}"; do
            if [[ "$phase" == "$name" || "$phase" == "$name"_* ]]; then
                printf '%s\n' "$phase"
                found=1
            fi
        done
        if [[ "$found" -eq 0 ]]; then
            echo "unknown phase: $name" >&2
            printf '  available: %s\n' "${all_phases[*]}" >&2
            exit 1
        fi
    done
}

if [[ "$action" == submit ]]; then
    for pair in ${STUDY_ENV:-}; do
        export "${pair?}"
    done
fi

# Match lib.sh without sourcing it (its phase functions redirect output).
if [[ "$action" != help && "$action" != probe ]]; then
    study_base="${STUDY_BASE:-$root/build/study/$(python3 "$root/scripts/study/source_id.py")}"
fi

case "$action" in

submit)
    command -v qsub > /dev/null || { echo "no qsub here: use 'local'" >&2; exit 1; }
    mapfile -t phases < <(resolve_phases "$@")

    # -V passes the whole environment to the job. PBS also has -v with a list
    # of pairs, but it does not survive values with spaces, and GRIDS="128 256"
    # has one. So it is enough to export beforehand:  GRIDS=128
    # ./scripts/run_study.sh submit On this site the PBS .o<jobid> file arrives
    # neither in the submission directory nor in the home. Telling it where to
    # put it is the only way of seeing a job that dies BEFORE opening its own
    # log -- which is exactly when one needs to see it.
    mkdir -p "$study_base/pbs"
    qsub_opts=(-V -o "$study_base/pbs/")
    [[ -n "${WALLTIME:-}" ]] && qsub_opts+=(-l "walltime=$WALLTIME")
    # The whole chain on one node. lib.sh repeats the constraint at every
    # resubmission: here it holds only for the first job.
    if [[ -n "${STUDY_HOST:-}" ]]; then
        qsub_opts+=(-l "select=1:ncpus=${STUDY_NCPUS:-112}:host=$STUDY_HOST")
    fi

    rejected=0
    for phase in "${phases[@]}"; do
        script="$phases_dir/$phase.sh"
        # A PBS refusal concerns a single phase -- usually a resource that that
        # queue does not grant -- and it is not a reason not to submit the
        # others. Before, it stopped here, and it looked as if everything had
        # failed.
        if ! id="$(qsub "${qsub_opts[@]}" "$script" 2>&1)"; then
            printf '  %-18s REJECTED: %s\n' "$phase" "$(head -1 <<< "$id")"
            rejected=$(( rejected + 1 ))
            continue
        fi
        printf '  %-18s %s\n' "$phase" "$id"
    done
    echo
    if [[ "$rejected" -gt 0 ]]; then
        echo "  $rejected phases rejected by PBS. Almost always it is the"
        echo "  walltime or the CPUs asked for: ./scripts/run_study.sh probe"
        echo "  says what each queue grants, and the #PBS directives at the top"
        echo "  of the script can be overridden from the command line"
        echo "  (qsub -l walltime=... script)."
        echo
    fi
    echo "  qstat -u \"\$USER\"    to follow them"
    echo "  the results land in $study_base/<phase>/results.csv"
    ;;

local)
    mapfile -t phases < <(resolve_phases "$@")
    for phase in "${phases[@]}"; do
        echo "### $phase"
        "$phases_dir/$phase.sh"
    done
    ;;

dry)
    mapfile -t phases < <(resolve_phases "$@")
    for phase in "${phases[@]}"; do
        echo "### $phase"
        DRY_RUN=1 "$phases_dir/$phase.sh"
    done
    ;;

merge)
    out="$study_base/all.csv"
    mkdir -p "$study_base"
    header=""
    intestazione=$(sed -n "s/^STUDY_HEADER='//p" "$root/scripts/study/lib.sh" \
        | head -1 | sed "s/'$//")
    colonne=$(awk -F, '{print NF}' <<< "$intestazione")
    : > "$out"
    for phase in "${all_phases[@]}"; do
        csv="$study_base/$phase/results.csv"
        [[ -f "$csv" ]] || continue
        if [[ -z "$header" ]]; then
            printf '%s\n' "$intestazione" > "$out"
            header=1
        fi
        # The columns added over time -- g_ms, then node -- all sit right
        # before status and note. A row of an older format has them missing,
        # and it is aligned by inserting them empty there: the plots read by
        # name, and a short row would shift every field for them.
        #
        # And a case retried with RETRY_FAILED=1 leaves two rows, the failed
        # one and the new one: the last counts, in place of the first. Columns
        # 1-13 (phase..steps) are enough to recognise a case: verified on the
        # dry runs of all six phases.
        awk -F, -v OFS=, -v larga="$colonne" '
        NR == 1 { next }
        {
            if (NF < larga) {
                # campi and not n: n counts the cases of the deduplication
                # below, and reusing it printed empty lines and lost one.
                campi = NF; stato = $(campi - 1); nota = $campi
                $larga = nota
                $(larga - 1) = stato
                for (i = campi - 1; i <= larga - 2; i++) $i = ""
            }
            id = $1
            for (i = 2; i <= 13; i++) id = id SUBSEP $i
            if (!(id in riga)) ordine[++n] = id
            riga[id] = $0
        }
        END { for (i = 1; i <= n; i++) print riga[ordine[i]] }' "$csv" >> "$out"
    done
    if [[ -z "$header" ]]; then
        echo "nothing to merge" >&2
        exit 1
    fi
    echo "$(( $(wc -l < "$out") - 1 )) measurements in $out"
    if [[ -x "$root/scripts/plot_matrix.py" ]]; then
        "$root/scripts/plot_matrix.py" "$out"
    fi
    ;;

probe)
    # What each queue really grants, measured instead of deduced: tiny jobs are
    # submitted and one looks at which is accepted. Those that pass are
    # cancelled immediately -- they must not run, only be accepted.
    command -v qsub > /dev/null || { echo "no qsub here" >&2; exit 1; }
    for queue in ${QUEUES:-scalability cpu}; do
        echo "queue $queue"
        printf '  cpus per job: '
        found=""
        for n in 112 56 28 14 7 1; do
            if id="$(echo /bin/true | qsub -q "$queue" -l "select=1:ncpus=$n" \
                     -l walltime=00:05:00 -N probe 2>&1)"; then
                found="$n"
                qdel "$id" > /dev/null 2>&1 || true
                break
            fi
        done
        if [[ -n "$found" ]]; then
            echo "select=1:ncpus=$found accepted"
        else
            echo "not even 1 cpu accepted -- $(head -1 <<< "${id:-}")"
            continue
        fi
        printf '  walltime:     '
        for w in 48:00:00 24:00:00 08:00:00 02:00:00 00:30:00 00:10:00; do
            if id="$(echo /bin/true | qsub -q "$queue" -l "select=1:ncpus=$found" \
                     -l "walltime=$w" -N probe 2>&1)"; then
                echo "$w accepted"
                qdel "$id" > /dev/null 2>&1 || true
                break
            fi
        done
    done
    echo
    echo "  The measuring phases want the whole node: if the exclusive queue"
    echo "  grants fewer CPUs than the node has, the central question of the"
    echo "  study (56 cores spent in different ways) cannot be asked."
    ;;

status)
    # The good width is that of lib.sh: a CSV started before g_ms existed also
    # has the old header, so comparing it with itself would say nothing.
    attesa=$(sed -n "s/^STUDY_HEADER='//p" "$root/scripts/study/lib.sh" \
        | head -1 | awk -F, '{print NF}')
    printf '  %-18s %8s %8s %8s   %s\n' phase cases ok failed updated
    for phase in "${all_phases[@]}"; do
        csv="$study_base/$phase/results.csv"
        if [[ ! -f "$csv" ]]; then
            printf '  %-18s %8s\n' "$phase" "-"
            continue
        fi
        # The status is always the second-to-last field and the note the last,
        # in this format as in the one before g_ms: the notes already have the
        # commas replaced, so counting from the right holds even on a file that
        # mixes the two widths. An index written by hand does not: it was $32,
        # g_ms moved it to $33, and for one night the study declared everything
        # failed. Every case counts with its last row: one retried with
        # RETRY_FAILED=1 has two, and the failed one no longer counts.
        read -r total ok bad vecchie < <(awk -F, -v larga="$attesa" '
            NR == 1 { next }
            {
                id = $1
                for (i = 2; i <= 13; i++) id = id SUBSEP $i
                stato[id] = $(NF - 1)
                corta[id] = (NF < larga)
            }
            END {
                for (id in stato) {
                    casi++
                    if (stato[id] == "ok") buoni++; else cattivi++
                    if (corta[id]) corte++
                }
                printf "%d %d %d %d\n", casi, buoni, cattivi, corte
            }' "$csv")
        [[ "$vecchie" -gt 0 ]] && nota=" ($vecchie in an older format)" || nota=""
        printf '  %-18s %8s %8s %8s   %s%s\n' "$phase" "$total" "$ok" "$bad" \
            "$(date -r "$csv" '+%Y-%m-%d %H:%M')" "$nota"
    done
    ;;

*)
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
esac
