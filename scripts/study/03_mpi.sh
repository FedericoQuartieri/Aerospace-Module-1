#!/usr/bin/env bash
#PBS -N nsb-03-mpi
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Fase 3 -- i processi da soli: un thread per rank, dominio diviso davvero.
#
# Domanda: a parita' di tutto il resto, quale dei due backend risolve meglio
# una linea spezzata, e da che punto in poi la divisione smette di rendere?
#
# E' il confronto che sul branch `pipeline' non si poteva fare: li' il
# pipelined Thomas era l'unico algoritmo compilato, e i 2.3 ms/passo citati per
# confronto venivano dal branch `mpi', cioe' da un altro albero, un'altra
# griglia e un altro insieme di flag (PIPELINE.md §4.2: "Nel repo non ci sono
# risultati di scaling"). Su `unified' i due backend sono lo stesso eseguibile
# a meno di -DTRIDIAG_*, e la differenza misurata e' solo loro.
#
# Le due taglie non sono ridondanti:
#
#   256^3   continuita' con le misure di MULTITHREAD.md §9. Con 7, 14, 28 o 56
#           rank la divisione non e' intera (256/7 = 36.6) e un blocco piu'
#           grande degli altri fa aspettare tutti: parte del tempo misurato e'
#           sbilanciamento.
#   224^3   224 = 7 x 32, quindi ogni conteggio di rank di questo studio la
#           divide esattamente. E' la riga pulita: se le due taglie raccontano
#           storie diverse, la differenza e' lo sbilanciamento.
#
# SIMD entra solo a 224^3, dove il confronto e' pulito: la domanda che pone e'
# quanto costa perdere i kernel vettorizzati sull'asse che viene diviso, e una
# taglia basta a rispondere.
#
#   qsub scripts/study/03_mpi.sh
#   GRIDS=224 qsub scripts/study/03_mpi.sh

# PBS non esegue questo file dove sta: ne mette una COPIA nella propria
# directory di spool e lancia quella. $BASH_SOURCE punta li', quindi non dice
# niente su dove sia il repo -- a dirlo e' PBS_O_WORKDIR, la directory da cui
# si e' fatto qsub. Cercare lib.sh accanto allo script funziona da riga di
# comando e fallisce dentro un job, prima ancora che esista un log in cui
# vederlo.
cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub va fatto dalla radice del repo; qui sono in $PWD" >&2
    exit 1
fi

source scripts/study/lib.sh
study_begin 03_mpi
study_machine

GRIDS="${GRIDS:-224 256}"
RANKS="${RANKS:-1 2 4 7 8 14 28 56}"
SIMD_GRID="${SIMD_GRID:-224}"

CASE_THREADS=1
# Build senza OpenMP: qui i thread non c'entrano, e la build a thread costa
# circa l'1% anche quando i thread sono uno (MULTITHREAD.md §7.3). Le righe
# ponte in fondo alla fase ricuciono con le fasi che invece lo usano.
CASE_OMP=0
CASE_MPI=1
CASE_REPEATS="${REPEATS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-900}"

for n in $GRIDS; do
    steps="${STEPS:-10}"
    [[ "$n" -le 128 ]] && steps="${STEPS_SMALL:-20}"

    echo "=== ${n}^3, $steps passi: rank crescenti, un thread ciascuno ==="
    for backend in schur pipeline; do
        for r in $RANKS; do
            study_case label="$backend ${n}^3 R=$r" backend="$backend" \
                ranks="$r" shape="$(study_auto_shape "$r")" \
                grid="$n $n $n" steps="$steps" simd=1
        done
    done

    if [[ "$n" == "$SIMD_GRID" ]]; then
        echo
        echo "=== ${n}^3 senza SIMD: quanto vale la vettorizzazione persa ==="
        for backend in schur pipeline; do
            for r in $RANKS; do
                study_case label="$backend ${n}^3 R=$r simd=0" \
                    backend="$backend" ranks="$r" \
                    shape="$(study_auto_shape "$r")" \
                    grid="$n $n $n" steps="$steps" simd=0
            done
        done
    fi
    echo
done

# Righe ponte: stessa configurazione, build con OpenMP e un thread solo. E'
# quanto va sottratto per confrontare questa fase con la 02 e la 05, che la
# usano.
echo "=== righe ponte: stessa cosa, ma build OpenMP a un thread ==="
for backend in schur pipeline; do
    for r in 1 56; do
        study_case label="$backend ponte R=$r omp=1" backend="$backend" \
            ranks="$r" shape="$(study_auto_shape "$r")" \
            grid="224 224 224" steps="${STEPS:-10}" simd=1 omp=1
    done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== risultato ==="
    awk -F, -v phase=03_mpi '
    NR == 1 || $1 != phase || $32 != "ok" || $2 ~ /ponte/ { next }
    {
        k = $3 "," $10 "," $5 "," $8
        wall[k] = $17; mpims[k] = $18; rss[k] = $29
        shape[k] = $14 "x" $15 "x" $16
        if (!($10 "," $5 in seen_g)) { gs[++ng] = $10 "," $5; seen_g[$10 "," $5] = 1 }
        if (!($8 in seen_r)) { rs[++nr] = $8; seen_r[$8] = 1 }
    }
    END {
        for (i = 1; i <= ng; i++) {
            split(gs[i], p, ",")
            printf "\n  %s^3  simd=%s\n", p[1], p[2]
            printf "  %-6s %-9s %28s %28s\n", "", "", "schur", "pipeline"
            printf "  %-6s %-9s %10s %8s %7s %10s %8s %7s\n",
                   "rank", "forma", "ms/passo", "speedup", "%mpi",
                   "ms/passo", "speedup", "%mpi"
            for (j = 1; j <= nr; j++) {
                r = rs[j]
                ka = "schur," gs[i] "," r; kb = "pipeline," gs[i] "," r
                if (!(ka in wall) && !(kb in wall)) continue
                if (!(gs[i] in ba) && (ka in wall)) ba[gs[i]] = wall[ka]
                if (!(gs[i] in bb) && (kb in wall)) bb[gs[i]] = wall[kb]
                printf "  %-6s %-9s", r, (ka in shape ? shape[ka] : shape[kb])
                if (ka in wall)
                    printf " %10.1f %7.2fx %6.0f%%", wall[ka], ba[gs[i]]/wall[ka],
                           100 * mpims[ka] / wall[ka]
                else printf " %10s %8s %7s", "-", "-", "-"
                if (kb in wall)
                    printf " %10.1f %7.2fx %6.0f%%", wall[kb], bb[gs[i]]/wall[kb],
                           100 * mpims[kb] / wall[kb]
                else printf " %10s %8s %7s", "-", "-", "-"
                printf "\n"
            }
        }
        print ""
        print "  `%mpi\x27 e\x27 la quota di passo passata dentro MPI. Schur la spende"
        print "  in collettive sulle interfacce, la pipeline in messaggi di batch:"
        print "  a parita\x27 di tempo totale, due modi diversi di essere lenti."
    }' "$STUDY_CSV"
fi

study_end
