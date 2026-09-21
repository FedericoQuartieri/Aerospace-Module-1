# shellcheck shell=bash
#
# The parts common to the phases of the scaling study.
#
# Every phase is an independent script, submittable on its own with qsub, that
# asks a single question. What they share lives here: how the machine is
# described, how a variant is compiled, how a case is run and how it is written
# to the CSV. A CSV row always has the same 33 columns, whichever phase
# produced it, so the results can be concatenated and compared without
# adapters.
#
# Three choices that hold for all the phases:
#
#   resume      the CSV is written one row at a time and every finished case
#               leaves its key in done.keys. If the walltime kills the job,
#               resubmitting it restarts from where it had got to instead of
#               redoing everything. FRESH=1 starts over.
#
#   timeout     every case has a time ceiling. A configuration that stalls --
#               and in the mixed zone some come close to it -- does not take
#               the rest of the job with it: it leaves a row with status
#               `timeout` and one goes on.
#
#   cached binaries   compiling is the only thing one does not want to repeat:
#               a variant (backend, simd, omp, mpi, batch) is built once per
#               phase and stays in $STUDY_BASE/variants/<phase>.
#
#   budget      the `scalability' queue grants 30 minutes per job
#               (resources_max.walltime = 00:30:00) and the study wants many
#               more. Every phase therefore works on a budget: when the useful
#               time is over it stops BEFORE being killed -- so the case in
#               progress is not cut in half -- and resubmits itself to
#               continue. An expired walltime loses nothing and requires no
#               intervention.

set -euo pipefail

STUDY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Results and resume keys belong to the sources AND the toolchain that made
# them. A new binary must never inherit successful cases from an old solver.
STUDY_STAMP="$(python3 "$STUDY_ROOT/scripts/study/source_id.py")"
STUDY_BASE="${STUDY_BASE:-$STUDY_ROOT/build/study/$STUDY_STAMP}"
STUDY_BIN="$STUDY_BASE/bin"

# Columns, once and for all: the rest of the file refers to this row. g_ms is
# at the end, after l2_p, and not next to eta_ms where its place would be: the
# summary awks of the phases read the columns by POSITION, and inserting it in
# the middle would shift them all. At the end it shifts two, status and note,
# and it is a one-line change per phase.
#
# g_ms is the time spent preparing the right-hand side of the eta step. It
# lives INSIDE eta_ms, not beside it: eta_ms minus g_ms is the pure solver. It
# must not be added to the other stages, or the step comes out longer than it
# is.
STUDY_HEADER='phase,label,backend,batch,simd,omp,mpi,ranks,threads,nx,ny,nz,steps,px,py,pz,wall_ms,mpi_ms,eta_ms,zeta_ms,u_ms,psi_ms,philow_ms,phihigh_ms,pressure_ms,porosity_ms,untimed_ms,cellstep_1e8s,rss_mb,l2_ux,l2_p,g_ms,node,status,note'

# How many fields the reading awk produces: it is needed to fill the row of a
# failed case with blanks without shifting the columns.
STUDY_MEASURED_FIELDS=19

# --------------------------------------------------------------- phase start

study_begin()
{
    local identity="$STUDY_BASE/source-toolchain.id"
    mkdir -p "$STUDY_BASE"
    if [[ ! -f "$identity" ]]; then
        if compgen -G "$STUDY_BASE/*/results.csv" > /dev/null; then
            echo "STUDY_BASE contains results without source identity; choose a new directory" >&2
            return 2
        fi
        # Six phases may start together. Publish complete contents only, and
        # never overwrite an identity installed concurrently by another job.
        local identity_tmp
        identity_tmp="$(mktemp "$STUDY_BASE/.source-id.XXXXXX")"
        printf '%s\n' "$STUDY_STAMP" > "$identity_tmp"
        ln "$identity_tmp" "$identity" 2>/dev/null || true
        rm -f "$identity_tmp"
    fi
    if [[ "$(cat "$identity")" != "$STUDY_STAMP" ]]; then
        echo "STUDY_BASE belongs to different sources/toolchain; choose a new directory" >&2
        return 2
    fi
    STUDY_PHASE="$1"
    STUDY_OUT="$STUDY_BASE/$STUDY_PHASE"
    STUDY_CSV="$STUDY_OUT/results.csv"
    STUDY_KEYS="$STUDY_OUT/done.keys"
    # RETRY_FAILED=1 puts the cases that failed back into play only once, here,
    # in the first job: their keys disappear (with a copy alongside) and from
    # that moment they are cases to do like the others. Ignoring them at every
    # read is not enough: study_resubmit passed the environment to the next
    # job, a real timeout recorded in this job was retried in every job of the
    # chain, and a case retried but postponed found the old key again and got
    # lost.
    if [[ "${RETRY_FAILED:-0}" == "1" && "${DRY_RUN:-0}" != "1" &&
          -f "$STUDY_KEYS" ]] && grep -q '^!' "$STUDY_KEYS"; then
        local back_in_play
        back_in_play=$(grep -c '^!' "$STUDY_KEYS")
        cp "$STUDY_KEYS" "$STUDY_KEYS.before-retry"
        grep -v '^!' "$STUDY_KEYS" > "$STUDY_KEYS.tmp" || true
        mv "$STUDY_KEYS.tmp" "$STUDY_KEYS"
        printf 'RETRY_FAILED=1: %d failed cases back in play (old keys in %s)\n' \
            "$back_in_play" "$STUDY_KEYS.before-retry"
    fi
    STUDY_LOG="$STUDY_OUT/run.log"
    STUDY_STARTED="$(date +%s)"
    STUDY_CASES=0
    STUDY_SKIPPED=0
    STUDY_FAILED=0
    STUDY_PENDING=0
    STUDY_OUT_OF_TIME=0
    mkdir -p "$STUDY_OUT" "$STUDY_BIN"

    if [[ "${FRESH:-0}" == "1" ]]; then
        rm -f "$STUDY_CSV" "$STUDY_KEYS" "$STUDY_OUT/chain.count"
        echo "FRESH=1: starting over, previous results are deleted"
        # And only for this job: the resubmission passes the environment with
        # `qsub -V', and an inherited FRESH would make every link of the chain
        # delete what the previous link has just measured.
        export FRESH=0
    fi

    # Useful work per job. The rest of the walltime is for compilation, the
    # summary and the margin to finish the case in progress without being
    # killed halfway: a measurement truncated by the walltime does not end up
    # in the CSV, and the time spent producing it is lost.
    STUDY_BUDGET="${STUDY_BUDGET:-1500}"
    # The number of the job in the chain is kept in a file, not in the
    # environment: passing it with `qsub -v STUDY_CHAIN=n+1' together with
    # `-V', the latter re-exported the old value over the new one and the
    # counter stayed stuck at 2 forever -- so the ceiling never triggered and
    # the chain never ended. A file, read after FRESH, does not have this
    # problem.
    STUDY_CHAIN=$(( $(cat "$STUDY_OUT/chain.count" 2> /dev/null || echo 0) + 1 ))
    echo "$STUDY_CHAIN" > "$STUDY_OUT/chain.count"
    STUDY_CHAIN_MAX="${STUDY_CHAIN_MAX:-40}"
    # The phases resubmit themselves until they are done, except whoever sets
    # this to 0 because it either succeeds in a minute or does not succeed at
    # all.
    STUDY_CHAINABLE="${STUDY_CHAINABLE:-1}"
    [[ -f "$STUDY_CSV" ]] || printf '%s\n' "$STUDY_HEADER" > "$STUDY_CSV"
    touch "$STUDY_KEYS"

    # On this site PBS's .o<jobid> arrives neither in the submission directory
    # nor in the home: the log must be kept next to the CSV, and in append mode
    # because a resume must not delete the previous one.
    exec > >(tee -a "$STUDY_LOG") 2>&1

    echo "==============================================================="
    echo "phase $STUDY_PHASE -- $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==============================================================="
    echo "results: $STUDY_CSV"
    printf 'budget:    %d min of useful work (job %d of the chain)\n' \
        $(( STUDY_BUDGET / 60 )) "$STUDY_CHAIN"
    [[ "${DRY_RUN:-0}" == "1" ]] && echo "DRY_RUN=1: listing the cases, not running them"
    echo
}

study_end()
{
    local elapsed=$(( $(date +%s) - STUDY_STARTED ))

    echo
    echo "=== end of phase $STUDY_PHASE ==="
    printf 'cases run: %d   skipped (already done): %d   failed: %d\n' \
        "$STUDY_CASES" "$STUDY_SKIPPED" "$STUDY_FAILED"
    printf 'total time: %02d:%02d:%02d\n' \
        $(( elapsed / 3600 )) $(( elapsed % 3600 / 60 )) $(( elapsed % 60 ))
    echo "csv: $STUDY_CSV"
    echo "log: $STUDY_LOG"

    if [[ "$STUDY_OUT_OF_TIME" -eq 0 ]]; then
        [[ "${DRY_RUN:-0}" == "1" ]] || echo "the phase is complete."
        return 0
    fi

    printf '%d cases left: this job has run out of budget.\n' \
        "$STUDY_PENDING"

    # If this job has not finished even one case, the next one would behave
    # identically: same queue, same first case, same outcome. Better to stop
    # and say so than to repeat the cycle -- it is exactly how two cases in
    # timeout occupied 31 jobs in a row.
    if [[ "$STUDY_CASES" -eq 0 && "$STUDY_FAILED" -eq 0 ]]; then
        echo "this job completed no case: stopping instead of repeating"
        echo "the same loop. Read the log above and resubmit by hand"
        echo "once you know why."
        return 0
    fi
    study_resubmit
}

# Continue on our own, instead of asking someone to resubmit every half hour.
# The chain number travels with the job and limits it: if something goes wrong
# in a repeatable way, the study stops by itself instead of filling the queue.
study_resubmit()
{
    local script="$STUDY_ROOT/scripts/study/$STUDY_PHASE.sh"

    if [[ "${AUTO_RESUBMIT:-$STUDY_CHAINABLE}" != "1" ]]; then
        echo "resubmission disabled: resubmit with  qsub $script"
        return 0
    fi
    if [[ "$STUDY_CHAIN" -ge "$STUDY_CHAIN_MAX" ]]; then
        echo "the chain reached $STUDY_CHAIN jobs: stopping here to be safe."
        echo "if that is expected, resubmit with  STUDY_CHAIN=1 qsub $script"
        return 0
    fi
    if ! command -v qsub > /dev/null || [[ -z "${PBS_JOBID:-}" ]]; then
        echo "outside PBS: rerun with  ./scripts/study/$STUDY_PHASE.sh"
        return 0
    fi

    local next pin=()
    mkdir -p "$STUDY_BASE/pbs"
    # A chain pinned to a node stays there. The `submit' constraint holds only
    # for the first job: without repeating it here, from the second link on the
    # chain would go to the first free node.
    if [[ -n "${STUDY_HOST:-}" ]]; then
        pin=(-l "select=1:ncpus=${STUDY_NCPUS:-112}:host=$STUDY_HOST")
    fi
    # Without RETRY_FAILED: the failed ones were already put back into play by
    # the first job of the chain, and a real timeout recorded here the next job
    # does not redo.
    if next="$(env -u RETRY_FAILED qsub -V ${pin[@]+"${pin[@]}"} -o "$STUDY_BASE/pbs/" "$script" 2>&1)"; then
        echo "continues in job $next"
    else
        echo "resubmission failed: $next"
        echo "resubmit by hand with  qsub $script"
    fi
}

# ------------------------------------------------------------------- machine

study_machine()
{
    STUDY_SOCKETS=$(lscpu | awk -F: '/^Socket\(s\)/ {gsub(/ /,"",$2); print $2}')
    STUDY_PER_SOCKET=$(lscpu | awk -F: '/^Core\(s\) per socket/ {gsub(/ /,"",$2); print $2}')
    : "${STUDY_SOCKETS:=1}" "${STUDY_PER_SOCKET:=1}"

    # How many CPUs the node has, and how many it gave to us. They are not the
    # same thing and telling them apart is everything:
    #
    #   nproc --all      the CPUs of the node, always, whatever they gave us
    #   NCPUS            those that PBS assigned to this job
    #   nproc            is NOT reliable here: it respects OMP_NUM_THREADS,
    #                    which PBS sets to the number of CPUs of the chunk,
    #                    and on a 112-CPU node with 7 assigned it answered 7
    #                    -- the right number by chance and for the wrong
    #                    reason.
    #   Cpus_allowed     on this cluster it is 0-111 even when the CPUs
    #                    granted are 7: the affinity mask is not restricted,
    #                    so by itself it does not say whether the node is
    #                    ours.
    local node_logical="$(nproc --all)"
    STUDY_LOGICAL="${NCPUS:-$node_logical}"
    STUDY_PHYSICAL=$(( STUDY_SOCKETS * STUDY_PER_SOCKET ))
    [[ "$STUDY_LOGICAL" -lt "$STUDY_PHYSICAL" ]] && STUDY_PHYSICAL="$STUDY_LOGICAL"

    echo "=== machine ==="
    printf 'node:          %s\n' "$(hostname -s)"
    printf 'sockets:       %s\n' "$STUDY_SOCKETS"
    printf 'node cpus:     %s logical, %s physical cores\n' \
        "$node_logical" "$(( STUDY_SOCKETS * STUDY_PER_SOCKET ))"
    printf 'cpus granted:  %s\n' "$STUDY_LOGICAL"
    printf 'NUMA nodes:    %s\n' \
        "$(find /sys/devices/system/node -maxdepth 1 -name 'node[0-9]*' 2>/dev/null | wc -l)"
    printf 'memory:        %s\n' "$(awk '/MemTotal/ {printf "%.0f GB", $2/1048576}' /proc/meminfo)"
    printf 'mpirun:        %s\n' "$(command -v "${MPIRUN:-mpirun}" || echo missing)"
    printf 'MPI version:   %s\n' \
        "$(${MPIRUN:-mpirun} --version 2>&1 | head -1)"
    if [[ -n "${STUDY_HOST:-}" ]]; then
        if [[ "$(hostname -s)" == "$STUDY_HOST" ]]; then
            printf 'pinned to:     %s\n' "$STUDY_HOST"
        else
            printf '\n  WARNING: STUDY_HOST=%s, but this job runs on %s.\n\n' \
                "$STUDY_HOST" "$(hostname -s)"
        fi
    fi

    # Is the node really all ours? On this cluster the answer was no two times
    # out of three, and the measurements collected in those cases were unusable
    # without anything in the output saying so.
    if [[ "$STUDY_LOGICAL" -ge "$node_logical" ]]; then
        STUDY_EXCLUSIVE=1
        printf 'exclusive:     yes\n'
    else
        STUDY_EXCLUSIVE=0
        printf '\n  WARNING: you have %s CPUs out of %s. The measurements below\n' \
            "$STUDY_LOGICAL" "$node_logical"
        printf '  are polluted by the neighbours\x27 jobs and must not be used for timings.\n'
        printf '  You need:  qsub -q scalability -l select=1:ncpus=%s\n\n' \
            "$node_logical"
    fi
    echo
}

# ---------------------------------------------------------------- compilation
#
# The Makefile is the only source of truth on the flags: here only the
# variables it documents are passed, and the binary produced is moved into the
# cache under the name of the variant. This way a difference between the study
# and the normal build cannot arise from flags copied by hand diverging.

# A fingerprint of the sources, computed once per job. It goes into the key of
# the binary cache: without it, an executable compiled from a different
# revision stays there and the campaign measures code that is not the current
# one. It happened, and on a large scale: 1706 cases out of 1888 of the first
# campaign were measured with the binaries of the previous campaign,
# recognisable because they do not print the `g term' line. The keys carried no
# trace of it.
study_source_stamp()
{
    printf '%s' "$STUDY_STAMP"
}

study_build()
{
    local backend="$1" simd="$2" omp="$3" mpi="$4" batch="$5"
    local target="${6:-bench}"
    local lines="$batch"
    [[ "$batch" == auto ]] && lines=""
    # One build tree per phase. The Makefile already separates the
    # configurations, but two phases that start together compile the same
    # configuration into the same objects, and an object half-written by one
    # job is linked by the other. The chains of a phase are queued one after
    # the other, so within a phase the cache stays.
    local options=(TRIDIAG="$backend" SIMD="$simd" OMP="$omp" MPI="$mpi"
                   PIPELINE_BATCH_LINES="$lines"
                   BUILD_ROOT="$STUDY_BASE/variants/$STUDY_PHASE")
    if [[ "$mpi" == 1 && -n "${MPICC:-}" ]]; then
        options+=(CC="$MPICC")
    elif [[ "$mpi" == 0 && -n "${CC:-}" ]]; then
        options+=(CC="$CC")
    fi
    local directory
    directory="$(make -s --no-print-directory -C "$STUDY_ROOT" "${options[@]}" print-test-dir)" || return 1
    local key="$target-$backend-simd$simd-omp$omp-mpi$mpi-b$batch-$STUDY_STAMP"
    local rebuild=()
    [[ "${REBUILD:-0}" == 1 ]] && rebuild=(-B)
    # Build directly at the configuration path; no shared bench followed by mv.
    if ! make -s --no-print-directory ${rebuild[@]+"${rebuild[@]}"} -C "$STUDY_ROOT" \
            "${options[@]}" "$directory/$target" > "$STUDY_BIN/$key.build.log" 2>&1; then
        echo "build failed: $key" >&2
        head -20 "$STUDY_BIN/$key.build.log" >&2
        return 1
    fi
    if [[ "$directory" == /* ]]; then
        printf '%s' "$directory/$target"
    else
        printf '%s' "$STUDY_ROOT/$directory/$target"
    fi
}

# The configuration file of a size: grid and steps, nothing else.
study_config()
{
    local nx="$1" ny="$2" nz="$3" steps="$4"
    local path="$STUDY_OUT/grid-${nx}x${ny}x${nz}-s${steps}.txt"

    printf 'width = %s\nheight = %s\ndepth = %s\nsteps = %s\nt_end = 1e-1\n' \
        "$nx" "$ny" "$nz" "$steps" > "$path"
    printf '%s' "$path"
}

# ----------------------------------------------------------------- placement
#
# Half of the result, and the half that no output reports when it is wrong. By
# default mpirun pins every process to a single core and its threads share it
# out: the threads column then measures zero gain, and it is an error that
# looks like a result (MULTITHREAD.md §8.1).

study_placement()
{
    local ranks="$1" threads="$2"

    : "${STUDY_SOCKETS:=1}" "${STUDY_LOGICAL:=$(nproc)}" "${STUDY_PHYSICAL:=1}"

    # Beyond the physical cores one enters SMT, and there the Open MPI count
    # changes: `PE=n' asks for n *cpus*, and by default a cpu is a core. With
    # 56 physical cores, 2 ranks x 56 threads ask for 112 and the launch is
    # refused ("binding more processes than cpus on a resource"): that is how
    # all the configurations with product 112 failed. With --use-hwthread-cpus
    # a cpu becomes a hardware thread, and the 112 are there.
    local units=$(( ranks * threads ))
    local smt=()
    local bind_unit=core
    if [[ "$units" -gt "$STUDY_PHYSICAL" && "$units" -le "$STUDY_LOGICAL" ]]; then
        smt=(--use-hwthread-cpus)
        bind_unit=hwthread
    fi

    STUDY_MPI_OPTS=()
    STUDY_OMP_BIND="close"
    STUDY_OMP_WAIT="active"
    STUDY_NOTE=""

    if [[ "$ranks" -eq 1 ]]; then
        # A single rank must cover all the sockets, and it is OpenMP that
        # distributes it: --bind-to none takes MPI out of the way, spread sends
        # the threads to all the memory channels instead of those of a single
        # socket.
        STUDY_MPI_OPTS=(--bind-to none)
        STUDY_OMP_BIND="spread"
    elif [[ "$threads" -eq 1 ]]; then
        STUDY_MPI_OPTS=("${smt[@]}" --map-by "$bind_unit" --bind-to "$bind_unit")
        # A single thread: OpenMP must not bind anything. Leaving
        # OMP_PROC_BIND=close with 56 ranks cost between 1.7x and 2.3x -- every
        # rank interprets OMP_PLACES on the global topology and re-binds its
        # main thread on top of it, undoing the placement that mpirun had just
        # made.
        STUDY_OMP_BIND="false"
    elif [[ $(( ranks % STUDY_SOCKETS )) -eq 0 ]]; then
        # One group of ranks per socket, and the threads of each inside its own
        # socket: without this the threads of different ranks mix across the
        # two sockets and one measures that disaster instead of the divided
        # domain.
        STUDY_MPI_OPTS=("${smt[@]}"
                        --map-by "ppr:$(( ranks / STUDY_SOCKETS )):socket:PE=$threads"
                        --bind-to "$bind_unit")
    else
        # A number of ranks that does not divide among the sockets cannot have
        # one group per socket. The answer is NOT --bind-to none: that way
        # every rank sees all the cores of the machine and spreads its threads
        # over them, those of different ranks end up on the same cores and
        # contend with each other while they spin in busy-wait. Measured: 7x4
        # on the pipeline gave 4708 ms with 93% of the time inside MPI, and the
        # two corresponding schur configurations did not finish at all.
        #
        # `slot:PE=n' gives each rank n distinct cores whatever the number of
        # ranks: one loses NUMA locality, not one's sanity.
        STUDY_MPI_OPTS=("${smt[@]}" --map-by "slot:PE=$threads" --bind-to "$bind_unit")
        STUDY_NOTE="$ranks rank non si dividono fra $STUDY_SOCKETS socket: niente localita' NUMA"
    fi

    if [[ $(( ranks * threads )) -gt "$STUDY_LOGICAL" ]]; then
        # More units than cpus: mpirun refuses to bind the processes to the
        # cores and would fail before starting. The case is not comparable with
        # the others anyway, and the note in the CSV says so.
        STUDY_MPI_OPTS=(--oversubscribe --bind-to none)
        STUDY_NOTE="${STUDY_NOTE:+$STUDY_NOTE; }in sovrannumero, senza binding"
        # With more threads than cores, the busy-waiting of OpenMP burns the
        # cores by contending them with whoever is working: an oversubscribed
        # case can slow down by orders of magnitude instead of linearly.
        # `passive' makes them sleep. It holds only here: where the measurement
        # matters, busy-waiting is the right one.
        STUDY_OMP_WAIT="passive"
    fi

    # A phase that knows better how the processes should be placed --
    # multi-node, where a hostfile and a per-node mapping are needed --
    # replaces everything.
    if [[ -n "${PLACEMENT_OVERRIDE:-}" ]]; then
        STUDY_MPI_OPTS=($PLACEMENT_OVERRIDE)
        STUDY_NOTE="${STUDY_NOTE:+$STUDY_NOTE; }piazzamento imposto dalla fase"
    fi
}

# ------------------------------------------------------------------ one case
#
# study_case backend=schur ranks=8 threads=1 grid="256 256 256" shape="2 2 2"
#
# Arguments in the form key=value, all optional: those not given come from the
# defaults of the phase (CASE_*). Unknown keys stop the script, because a
# misspelt parameter is a wrong result.

study_case()
{
    local backend="${CASE_BACKEND:-schur}"
    # auto: the binary does not fix the batch and the pipeline chooses it at
    # start-up, as the solver compiled without options does. A number fixes it
    # at compile time, and it is what the scan of phase 13 needs. Before, the
    # default was 64, and the phases that do not sweep the batch measured a
    # value the code no longer uses.
    local batch="${CASE_BATCH:-auto}"
    local simd="${CASE_SIMD:-1}"
    local omp="${CASE_OMP:-1}"
    local mpi="${CASE_MPI:-1}"
    local ranks="${CASE_RANKS:-1}"
    local threads="${CASE_THREADS:-1}"
    local grid="${CASE_GRID:-128 128 128}"
    local shape="${CASE_SHAPE:-}"
    local steps="${CASE_STEPS:-10}"
    local repeats="${CASE_REPEATS:-${REPEATS:-2}}"
    local norms="${CASE_NORMS:-0}"
    local label=""
    local note=""
    local wrap=""
    local bind=""
    local arg

    for arg in "$@"; do
        case "$arg" in
            backend=*) backend="${arg#*=}" ;;
            batch=*)   batch="${arg#*=}" ;;
            simd=*)    simd="${arg#*=}" ;;
            omp=*)     omp="${arg#*=}" ;;
            mpi=*)     mpi="${arg#*=}" ;;
            ranks=*)   ranks="${arg#*=}" ;;
            threads=*) threads="${arg#*=}" ;;
            grid=*)    grid="${arg#*=}" ;;
            shape=*)   shape="${arg#*=}" ;;
            steps=*)   steps="${arg#*=}" ;;
            repeats=*) repeats="${arg#*=}" ;;
            norms=*)   norms="${arg#*=}" ;;
            label=*)   label="${arg#*=}" ;;
            note=*)    note="${arg#*=}" ;;
            wrap=*)    wrap="${arg#*=}" ;;
            bind=*)    bind="${arg#*=}" ;;
            *) echo "study_case: unknown argument: $arg" >&2; return 2 ;;
        esac
    done

    local nx ny nz
    read -r nx ny nz <<< "$grid"
    [[ -z "$label" ]] && label="$backend/r${ranks}t${threads}/${nx}"

    if [[ -n "${STUDY_EXPECTED:-}" ]]; then
        printf '%s\t%s\n' \
            "$label|$backend|$batch|$simd|$omp|$mpi|$ranks|$threads|$nx|$ny|$nz|$steps" \
            "$shape" >> "$STUDY_EXPECTED"
    fi

    # The resume key contains everything that changes the measurement: two
    # cases with the same key are the same case.
    local key="$STUDY_PHASE|$label|$backend|$batch|$simd|$omp|$mpi|$ranks|$threads|$nx|$ny|$nz|$steps|${shape// /-}|${wrap// /-}|$bind"

    if [[ "${RESUME:-1}" == "1" ]] && study_already_done "$key"; then
        STUDY_SKIPPED=$(( STUDY_SKIPPED + 1 ))
        printf '  %-34s already done\n' "$label"
        return 0
    fi

    # The remaining time decides whether this case is started. Starting one
    # that will not finish in time means being killed by the walltime halfway:
    # the measurement does not end up in the CSV and the time is thrown away.
    local remaining=$(( STUDY_BUDGET - ( $(date +%s) - STUDY_STARTED ) ))
    if [[ "${DRY_RUN:-0}" != "1" && "$remaining" -lt "${CASE_MIN_TIME:-90}" ]]; then
        if [[ "$STUDY_OUT_OF_TIME" -eq 0 ]]; then
            printf '  budget spent: the remaining cases go to the next job\n'
            STUDY_OUT_OF_TIME=1
        fi
        STUDY_PENDING=$(( STUDY_PENDING + 1 ))
        return 0
    fi

    printf '  %-34s ' "$label"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf 'backend=%s batch=%s simd=%s omp=%s mpi=%s ranks=%s thr=%s grid=%sx%sx%s shape=%s steps=%s\n' \
            "$backend" "$batch" "$simd" "$omp" "$mpi" "$ranks" "$threads" \
            "$nx" "$ny" "$nz" "${shape:-auto}" "$steps"
        STUDY_CASES=$(( STUDY_CASES + 1 ))
        return 0
    fi

    local exe config
    if ! exe="$(study_build "$backend" "$simd" "$omp" "$mpi" "$batch")"; then
        study_record "$label" "$backend" "$batch" "$simd" "$omp" "$mpi" \
            "$ranks" "$threads" "$nx" "$ny" "$nz" "$steps" "" \
            "build" "build failed"
        printf 'build failed\n'
        STUDY_FAILED=$(( STUDY_FAILED + 1 ))
        return 0
    fi
    config="$(study_config "$nx" "$ny" "$nz" "$steps")"

    study_placement "$ranks" "$threads"
    # bind= forces the placement of the threads when that is exactly what is
    # being studied: the same threads on one socket or on two are the same
    # computing power and half or all of the memory channels.
    [[ -n "$bind" ]] && STUDY_OMP_BIND="$bind"
    [[ -n "$STUDY_NOTE" ]] && note="${note:+$note; }$STUDY_NOTE"

    local best_wall="" best_line="" status="ok" started
    local repeat out line wall chosen=""
    started="$(date +%s)"

    # No case can last longer than the budget that remains: beyond that the
    # walltime would kill it anyway, without leaving any trace. case_timeout is
    # already cut by the budget, phase_timeout is not: it is with that one that
    # a case that is too slow is told apart from one that started late.
    local phase_timeout="${CASE_TIMEOUT:-1500}"
    local case_timeout="$phase_timeout"
    [[ "$remaining" -lt "$case_timeout" ]] && case_timeout="$remaining"

    # A binary without MPI is not launched with mpirun: it is just executed,
    # pinned to a single core because the serial case is a reference and must
    # not migrate between the sockets in the middle of the measurement.
    local -a command
    if [[ "$mpi" == "0" ]]; then
        command=()
        if [[ "$threads" -eq 1 ]] && command -v taskset > /dev/null; then
            command=(taskset -c 0)
        fi
        [[ -n "$wrap" ]] && command+=($wrap)
        command+=("$exe" "$config")
    else
        command=("${MPIRUN:-mpirun}" "${STUDY_MPI_OPTS[@]}"
                 -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND
                 -x OMP_WAIT_POLICY -x BENCH_NORMS
                 -n "$ranks")
        # The wrapper sits between mpirun and the executable: it is every rank
        # that must inherit its constraints, not the launcher.
        [[ -n "$wrap" ]] && command+=($wrap)
        command+=("$exe" "$config")
        [[ -n "$shape" ]] && command+=($shape)
    fi

    local left repeat_timeout repeat_started last_seconds=0
    for (( repeat = 0; repeat < repeats; repeat++ )); do
        # The budget is re-checked before every repetition. Before, it was
        # computed once per case and every repetition received the whole
        # case_timeout: two repetitions could last twice what remained, the
        # walltime killed the job halfway through a case, and the resubmission
        # -- which is at the bottom of the script -- did not start. That is how
        # phase 13 stopped at 374 cases out of 440 with the queue empty.
        left=$(( STUDY_BUDGET - ( $(date +%s) - STUDY_STARTED ) ))
        if [[ "$repeat" -gt 0 && "$left" -le "$last_seconds" ]]; then
            # Another repetition would not fit. The measurement that exists is
            # valid: it is kept, and the note says how many repetitions it is
            # taken over.
            note="${note:+$note; }ripetizioni: $repeat su $repeats (budget)"
            break
        fi
        repeat_timeout="$case_timeout"
        [[ "$left" -lt "$repeat_timeout" ]] && repeat_timeout="$left"
        repeat_started="$(date +%s)"

        set +e
        # stdin closed: mpirun reads it and consumes it, and a phase that
        # generates the cases with `while read ... done < <(...)' would see the
        # list vanish after the first case. It happened: phase 12 measured 6
        # cases out of 276 and declared itself complete.
        out="$(OMP_NUM_THREADS="$threads" \
               OMP_PLACES="${OMP_PLACES:-cores}" \
               OMP_PROC_BIND="$STUDY_OMP_BIND" \
               OMP_WAIT_POLICY="$STUDY_OMP_WAIT" \
               BENCH_NORMS="$norms" \
               timeout --kill-after=30 "$repeat_timeout" \
               "${command[@]}" 2>&1 < /dev/null)"
        local code=$?
        set -e
        last_seconds=$(( $(date +%s) - repeat_started ))

        if [[ $code -eq 124 || $code -eq 137 ]]; then
            # Cut by the budget and not by its own timeout, with a measurement
            # already taken: the configuration is the same, and that
            # measurement stays.
            if [[ -n "$best_line" && "$repeat_timeout" -lt "$phase_timeout" ]]; then
                note="${note:+$note; }ripetizioni: $repeat su $repeats (budget)"
                break
            fi
            # Cut by the budget at the first repetition: it is not known how
            # long it lasts, it is only known that it did not fit in what
            # remained. Writing it as a timeout would mark it as failed
            # forever, so it is postponed: the next job gets to it first, with
            # the whole budget. Only if this job has already finished
            # something, though: the first case of a job had the maximum that a
            # job can give, and for it timeout is the true answer. Otherwise a
            # case longer than a whole job would be postponed forever.
            if [[ "$repeat_timeout" -lt "$phase_timeout" &&
                  $(( STUDY_CASES + STUDY_FAILED )) -gt 0 ]]; then
                status="deferred"
                break
            fi
            status="timeout"
            break
        fi
        if [[ $code -ne 0 ]]; then
            status="failed"
            printf '\n    exit %d, here is what it printed:\n' "$code"
            sed 's/^/      /' <<< "$out" | tail -15
            break
        fi

        line="$(study_parse <<< "$out")" || line=""
        if [[ -z "$line" ]]; then
            status="unreadable"
            printf '\n    no timings in the output:\n'
            sed 's/^/      /' <<< "$out" | tail -15
            break
        fi
        chosen="$(awk '/^  bench batch:/ { print $3; exit }' <<< "$out")"

        # px,py,pz are the first three fields: the time is the fourth.
        wall="$(cut -d, -f4 <<< "$line")"
        if [[ -z "$best_wall" ]] || awk "BEGIN{exit !($wall < $best_wall)}"; then
            best_wall="$wall"
            best_line="$line"
        fi
    done

    local seconds=$(( $(date +%s) - started ))

    # No row and no key: for the resume the case never existed.
    if [[ "$status" == "deferred" ]]; then
        STUDY_OUT_OF_TIME=1
        STUDY_PENDING=$(( STUDY_PENDING + 1 ))
        printf 'cut by the budget, back to the next job (%ds)\n' "$seconds"
        return 0
    fi

    if [[ "$status" != "ok" ]]; then
        STUDY_FAILED=$(( STUDY_FAILED + 1 ))
        study_record "$label" "$backend" "$batch" "$simd" "$omp" "$mpi" \
            "$ranks" "$threads" "$nx" "$ny" "$nz" "$steps" "" \
            "$status" "$note"
        # A case that went badly is also a finished case: without this row
        # every job of the chain retries it, spends the whole budget on the
        # same timeout and never gets to the cases after it. It happened: two
        # cases consumed 31 jobs in a row without making anything progress.
        # RETRY_FAILED=1 puts them back into play, when something has been
        # changed that might make them succeed.
        study_done "$key" "$status"
        printf '%s (%ds)\n' "$status" "$seconds"
        return 0
    fi

    # The batch column stays `auto', like the resume key and the expected list
    # of phase 15: it is the configuration. The value that the pipeline chose
    # is printed by bench, and goes into the note.
    if [[ "$batch" == auto && "$backend" == pipeline && -n "$chosen" ]]; then
        note="${note:+$note; }batch scelto $chosen"
    fi

    study_record "$label" "$backend" "$batch" "$simd" "$omp" "$mpi" \
        "$ranks" "$threads" "$nx" "$ny" "$nz" "$steps" "$best_line" \
        ok "$note"
    study_done "$key"
    STUDY_CASES=$(( STUDY_CASES + 1 ))

    local shape_out untimed rss
    shape_out="$(cut -d, -f1-3 <<< "$best_line" | tr ',' 'x')"
    untimed="$(cut -d, -f14 <<< "$best_line")"
    rss="$(cut -d, -f16 <<< "$best_line")"
    printf '%10s ms  %-9s unaccounted %6s ms  rss %7s MB  (%ds)\n' \
        "$best_wall" "$shape_out" "$untimed" "$rss" "$seconds"
}

# The two references from which the speedups of a configuration are normalised.
#
#   serial  MPI=0 OMP=0.  The binary does not link MPI and does not compile
#           OpenMP: the directives are empty macros and the MPI functions are
#           stubs that answer "one process, no neighbour".  It is not the
#           parallel code at one process, it is a binary in which the parallel
#           code does not exist.  It is the T_s of the absolute speedup.
#
#   T(1)    MPI=1 OMP=0, one rank.  The same source with the parallel
#           machinery compiled in, but a single process.
#
# Having both is not redundancy: their difference IS the measure of the cost of
# the idle parallel machinery.  As long as it stays within the noise, using
# T(1) as the denominator is legitimate, and it is demonstrated instead of
# asserted. Both at one thread, because the comparison must be between
# configurations identical in everything except the layer being measured.
#
# It takes the same arguments as study_case except omp, mpi, ranks and threads,
# which are fixed here: passing them would mean measuring something else, and
# indeed they stop the script.
study_baseline()
{
    local label="" arg
    local rest=()

    for arg in "$@"; do
        case "$arg" in
            label=*) label="${arg#*=}" ;;
            omp=*|mpi=*|ranks=*|threads=*|shape=*)
                echo "study_baseline: $arg is set by the function, do not pass it" >&2
                return 2 ;;
            *) rest+=("$arg") ;;
        esac
    done

    study_case "${rest[@]}" label="$label seriale" \
        omp=0 mpi=0 ranks=1 threads=1
    study_case "${rest[@]}" label="$label T(1)" \
        omp=0 mpi=1 ranks=1 threads=1
}

# Reads the output of bench. The patterns are anchored: "eta system" without an
# anchor would also match "zeta system", and the times of zeta would be read
# believing them to be those of eta -- plausible numbers, of something else
# (MULTITHREAD.md §8.3).
study_parse()
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
        /^  bench proc grid/   { px = $4; py = $6; pz = $8 }
        /^  bench peak rss/    { rss = $4 }
        /^  L2 error u_x/   { lux = $4 }
        /^  L2 error p/     { lp  = $4 }
        END {
            if (wall == "") { exit 1 }
            sum = eta + zeta + u + psi + lo + hi + pr + po
            # g does NOT enter the sum: it is inside eta, and adding it would
            # count the eta step twice.
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.3f,%s,%s,%s,%s,%s",
                   px, py, pz, wall, mpi, eta, zeta, u, psi, lo, hi, pr, po,
                   wall - sum, cell, rss, lux, lp, g
        }'
}

# A CSV row, always with the same number of columns even when the case produced
# nothing: a hole that shifts the columns is discovered weeks later.
study_record()
{
    local label="$1" backend="$2" batch="$3" simd="$4" omp="$5" mpi="$6"
    local ranks="$7" threads="$8" nx="$9" ny="${10}" nz="${11}" steps="${12}"
    local measured="${13}" status="${14}" note="${15}"

    if [[ -z "$measured" ]]; then
        measured="$(printf ',%.0s' $(seq 2 "$STUDY_MEASURED_FIELDS"))"
    fi

    # The node in every row: the nodes of the queue declare the same machine
    # and at full load they are not (cpu04 up to 1.7x slower than cpu03).
    # Without this column it could only be reconstructed by cross-checking
    # run.log line by line.
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$STUDY_PHASE" "$label" "$backend" "$batch" "$simd" "$omp" "$mpi" \
        "$ranks" "$threads" "$nx" "$ny" "$nz" "$steps" "$measured" \
        "$(hostname -s)" "$status" "${note//,/;}" >> "$STUDY_CSV"
}

# A finished case -- succeeded or not -- leaves its key, so the resume skips
# it. The keys of failed cases carry a marker, so that with RETRY_FAILED=1 they
# can be retried without redoing the successful ones too.
study_done()
{
    local marker=""
    [[ "${2:-ok}" == "ok" ]] || marker="!"
    printf '%s%s\n' "$marker" "$1" >> "$STUDY_KEYS"
}

study_already_done()
{
    grep -qxF "$1" "$STUDY_KEYS" && return 0
    [[ "${RETRY_FAILED:-0}" == "1" ]] && return 1
    grep -qxF "!$1" "$STUDY_KEYS"
}

# The two shapes of process grid that this study uses most often.
# MPI_Dims_create chooses them like this, and having them written here makes it
# possible to impose the same shape also where one wants it different from the
# default.
study_auto_shape()
{
    case "$1" in
        1)  echo "1 1 1" ;;
        2)  echo "2 1 1" ;;
        4)  echo "2 2 1" ;;
        7)  echo "7 1 1" ;;
        8)  echo "2 2 2" ;;
        14) echo "7 2 1" ;;
        16) echo "4 2 2" ;;
        28) echo "7 2 2" ;;
        56) echo "7 4 2" ;;
        112) echo "7 4 4" ;;
        *)  echo "" ;;
    esac
}
