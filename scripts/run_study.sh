#!/usr/bin/env bash
#
# Lo studio di scaling completo: dieci fasi, un comando.
#
#   ./scripts/run_study.sh submit          sottomette tutte le fasi a PBS
#   ./scripts/run_study.sh submit 03 05    solo alcune
#   ./scripts/run_study.sh local 00        esegue una fase qui e ora
#   ./scripts/run_study.sh dry             elenca i casi senza eseguirli
#   ./scripts/run_study.sh merge           unisce i CSV e disegna i grafici
#   ./scripts/run_study.sh status          a che punto sono le fasi
#   ./scripts/run_study.sh probe           cosa concede ogni coda, misurato
#
# La fase 00 va sempre per prima e da sola: verifica che tutte le varianti
# diano la stessa risposta -- se non e' cosi', i tempi delle altre fasi
# confrontano programmi diversi -- e riempie la cache dei binari, che le altre
# fasi condividono e che due job simultanei si contenderebbero. Le altre
# dipendono da lei con `-W depend=afterok' e possono poi correre in parallelo:
# ognuna chiede un nodo intero, quindi PBS non le mettera' mai sulla stessa
# macchina a disturbarsi.
#
# La coda `scalability' concede 30 minuti per job (resources_max.walltime =
# 00:30:00) e lo studio ne vuole molte di piu'. Non serve fare niente: ogni
# fase lavora a budget, smette prima di essere uccisa e si ri-sottomette da
# sola finche' non ha finito. `status' dice a che punto e'. Sottomettere di
# nuovo una fase gia' in corso non fa danni -- i casi gia' misurati vengono
# saltati -- quindi `submit' vale anche come "continua".
#
# Variabili utili, passate cosi':
#
#   GRIDS=128 REPEATS=1 ./scripts/run_study.sh submit 03
#   WALLTIME=12:00:00 ./scripts/run_study.sh submit 05
#
# Le variabili dell'ambiente arrivano al job con `qsub -V, quindi basta
# metterle davanti al comando. STUDY_ENV="chiave=valore ..." fa lo stesso per
# chi preferisce, ma i valori non possono contenere spazi.
#
#   FRESH=1     ricomincia da capo invece di riprendere
#   STUDY_BUDGET=1500  secondi di lavoro utile per job: 25 dei 30 minuti
#               concessi, il resto e' margine per chiudere il caso in corso
#   AUTO_RESUBMIT=0    non ri-sottomettersi, fermarsi a budget finito
#   DRY_RUN=1   elenca i casi e non esegue niente
#   REPEATS     ripetizioni per caso (default 2, si tiene la migliore)
#   STEPS       passi temporali per caso

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
phases_dir="$root/scripts/study"
action="${1:-help}"
shift || true

all_phases=(00_env 01_ceiling 02_threads 03_mpi 04_shape 05_hybrid \
            06_batch 07_weak 08_size 09_multinode)

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
            echo "fase sconosciuta: $name" >&2
            printf '  disponibili: %s\n' "${all_phases[*]}" >&2
            exit 1
        fi
    done
}

case "$action" in

submit)
    command -v qsub > /dev/null || { echo "qsub non c'e': usa 'local'" >&2; exit 1; }
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

    first=""
    rejected=0
    for phase in "${phases[@]}"; do
        script="$phases_dir/$phase.sh"
        deps=()
        # Tutte dipendono dalla 00: e' lei a dire se ha senso misurare, ed e'
        # lei a compilare le varianti che le altre si limitano a usare.
        if [[ "$phase" != 00_env && -n "$first" ]]; then
            deps=(-W "depend=afterok:$first")
        fi
        # Un rifiuto di PBS riguarda una fase sola -- di solito una risorsa
        # che quella coda non concede -- e non e' un motivo per non sottomettere
        # le altre. Prima si fermava qui, e sembrava che fosse fallito tutto.
        if ! id="$(qsub "${qsub_opts[@]}" "${deps[@]}" "$script" 2>&1)"; then
            printf '  %-14s RIFIUTATO: %s\n' "$phase" "$(head -1 <<< "$id")"
            rejected=$(( rejected + 1 ))
            continue
        fi
        printf '  %-14s %s\n' "$phase" "$id"
        [[ "$phase" == 00_env ]] && first="$id"
    done
    echo
    if [[ "$rejected" -gt 0 ]]; then
        echo "  $rejected fasi rifiutate da PBS. Quasi sempre e' il walltime o"
        echo "  le CPU chieste: ./scripts/run_study.sh probe  dice cosa concede"
        echo "  ogni coda, e le direttive #PBS in testa allo script si possono"
        echo "  scavalcare da riga di comando (qsub -l walltime=... script)."
        echo
    fi
    echo "  qstat -u \"\$USER\"    per seguirle"
    echo "  i risultati arrivano in build/study/<fase>/results.csv"
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
    : > "$out"
    for phase in "${all_phases[@]}"; do
        csv="$root/build/study/$phase/results.csv"
        [[ -f "$csv" ]] || continue
        if [[ -z "$header" ]]; then
            head -1 "$csv" > "$out"
            header=1
        fi
        tail -n +2 "$csv" >> "$out"
    done
    if [[ -z "$header" ]]; then
        echo "nessun risultato da unire" >&2
        exit 1
    fi
    echo "$(( $(wc -l < "$out") - 1 )) misure in $out"
    if [[ -x "$root/scripts/plot_study.py" ]]; then
        "$root/scripts/plot_study.py" "$out"
    fi
    ;;

probe)
    # Cosa concede davvero ogni coda, misurato invece che dedotto: si
    # sottomettono job minuscoli e si guarda quale viene accettato. Quelli che
    # passano vengono cancellati subito -- non devono girare, solo essere
    # accettati.
    command -v qsub > /dev/null || { echo "qsub non c'e'" >&2; exit 1; }
    for queue in ${QUEUES:-scalability cpu}; do
        echo "coda $queue"
        printf '  cpu per job:  '
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
            echo "select=1:ncpus=$found accettato"
        else
            echo "nemmeno 1 cpu accettata -- $(head -1 <<< "${id:-}")"
            continue
        fi
        printf '  walltime:     '
        for w in 48:00:00 24:00:00 08:00:00 02:00:00 00:30:00 00:10:00; do
            if id="$(echo /bin/true | qsub -q "$queue" -l "select=1:ncpus=$found" \
                     -l "walltime=$w" -N probe 2>&1)"; then
                echo "$w accettato"
                qdel "$id" > /dev/null 2>&1 || true
                break
            fi
        done
    done
    echo
    echo "  Le fasi che misurano vogliono il nodo intero: se la coda esclusiva"
    echo "  concede meno CPU di quelle del nodo, la domanda centrale dello"
    echo "  studio (56 core spesi in modi diversi) non si puo' porre."
    ;;

status)
    printf '  %-14s %8s %8s %8s   %s\n' fase misure ok falliti aggiornato
    for phase in "${all_phases[@]}"; do
        csv="$root/build/study/$phase/results.csv"
        if [[ ! -f "$csv" ]]; then
            printf '  %-14s %8s\n' "$phase" "-"
            continue
        fi
        total=$(( $(wc -l < "$csv") - 1 ))
        ok=$(awk -F, 'NR>1 && $32=="ok"' "$csv" | wc -l)
        bad=$(awk -F, 'NR>1 && $32!="ok"' "$csv" | wc -l)
        printf '  %-14s %8s %8s %8s   %s\n' "$phase" "$total" "$ok" "$bad" \
            "$(date -r "$csv" '+%Y-%m-%d %H:%M')"
    done
    ;;

*)
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
esac
