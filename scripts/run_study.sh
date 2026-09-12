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

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
phases_dir="$root/scripts/study"
action="${1:-help}"
shift || true

# Le sei fasi riempiono la matrice invece di rispondere a una domanda per
# volta: durano giorni e si ri-sottomettono da sole. Vedi
# scripts/study/matrix.sh. Lo studio mirato a dieci fasi che le precedeva e'
# stato tolto -- resta nella storia di git -- perche' quello che misurava lo
# misurano queste, per intero.
all_phases=(10_matrix_threads 11_matrix_mpi 12_matrix_hybrid \
            13_matrix_batch 14_matrix_size 15_matrix_check)

# Un prefisso numerico basta a scegliere una fase: `05' vale `05_hybrid'.
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

case "$action" in

submit)
    command -v qsub > /dev/null || { echo "no qsub here: use 'local'" >&2; exit 1; }
    mapfile -t phases < <(resolve_phases "$@")

    # -V passa tutto l'ambiente al job. PBS ha anche -v con una lista di
    # coppie, ma non sopravvive ai valori con spazi, e GRIDS="128 256" ne ha.
    # Cosi' basta esportare prima:  GRIDS=128 ./scripts/run_study.sh submit
    # Su questo sito il file .o<jobid> di PBS non arriva ne' nella directory di
    # sottomissione ne' nella home. Dirgli dove metterlo e' l'unico modo di
    # vedere un job che muore PRIMA di aprire il proprio log -- che e'
    # esattamente quando serve vederlo.
    mkdir -p "$root/build/study/pbs"
    qsub_opts=(-V -o "$root/build/study/pbs/")
    [[ -n "${WALLTIME:-}" ]] && qsub_opts+=(-l "walltime=$WALLTIME")
    for pair in ${STUDY_ENV:-}; do
        export "${pair?}"
    done

    rejected=0
    for phase in "${phases[@]}"; do
        script="$phases_dir/$phase.sh"
        # Un rifiuto di PBS riguarda una fase sola -- di solito una risorsa
        # che quella coda non concede -- e non e' un motivo per non sottomettere
        # le altre. Prima si fermava qui, e sembrava che fosse fallito tutto.
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
    echo "  the results land in build/study/<phase>/results.csv"
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
    out="$root/build/study/all.csv"
    mkdir -p "$root/build/study"
    header=""
    intestazione=$(sed -n "s/^STUDY_HEADER='//p" "$root/scripts/study/lib.sh" \
        | head -1 | sed "s/'$//")
    colonne=$(awk -F, '{print NF}' <<< "$intestazione")
    : > "$out"
    for phase in "${all_phases[@]}"; do
        csv="$root/build/study/$phase/results.csv"
        [[ -f "$csv" ]] || continue
        if [[ -z "$header" ]]; then
            printf '%s\n' "$intestazione" > "$out"
            header=1
        fi
        # Le misure fatte prima che g_ms esistesse hanno una colonna in meno.
        # Qui si allineano al formato di adesso -- g_ms vuoto, inserito prima
        # di stato e nota -- perche' i grafici leggono per nome e una riga
        # corta gli sposterebbe ogni campo di uno.
        awk -F, -v OFS=, -v larga="$colonne" 'NR > 1 {
            if (NF == larga - 1) {
                stato = $(NF - 1); nota = $NF
                $(NF - 1) = ""
                $NF = stato
                $(NF + 1) = nota
            }
            print
        }' "$csv" >> "$out"
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
    # Cosa concede davvero ogni coda, misurato invece che dedotto: si
    # sottomettono job minuscoli e si guarda quale viene accettato. Quelli che
    # passano vengono cancellati subito -- non devono girare, solo essere
    # accettati.
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
    # La larghezza buona e' quella di lib.sh: un CSV cominciato prima che g_ms
    # esistesse ha anche l'intestazione vecchia, quindi confrontarlo con se
    # stesso non direbbe niente.
    attesa=$(sed -n "s/^STUDY_HEADER='//p" "$root/scripts/study/lib.sh" \
        | head -1 | awk -F, '{print NF}')
    printf '  %-18s %8s %8s %8s   %s\n' phase cases ok failed updated
    for phase in "${all_phases[@]}"; do
        csv="$root/build/study/$phase/results.csv"
        if [[ ! -f "$csv" ]]; then
            printf '  %-18s %8s\n' "$phase" "-"
            continue
        fi
        total=$(( $(wc -l < "$csv") - 1 ))
        # Lo stato e' sempre il penultimo campo e la nota l'ultimo, in questo
        # formato come in quello prima di g_ms: le note hanno le virgole
        # gia' sostituite, quindi contare da destra regge anche su un file
        # che mescola le due larghezze. Un indice scritto a mano no: era $32,
        # g_ms l'ha spostato a $33, e per una notte lo studio ha dichiarato
        # fallito tutto quanto.
        read -r ok bad vecchie < <(awk -F, -v larga="$attesa" '
            NR == 1 { next }
            { if ($(NF - 1) == "ok") buoni++; else cattivi++
              if (NF < larga) corte++ }
            END { printf "%d %d %d\n", buoni, cattivi, corte }' "$csv")
        [[ "$vecchie" -gt 0 ]] && nota=" ($vecchie without g_ms)" || nota=""
        printf '  %-18s %8s %8s %8s   %s%s\n' "$phase" "$total" "$ok" "$bad" \
            "$(date -r "$csv" '+%Y-%m-%d %H:%M')" "$nota"
    done
    ;;

*)
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
esac
