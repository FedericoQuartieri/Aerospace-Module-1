#!/usr/bin/env bash
#PBS -N nsb-08-size
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=06:00:00
#PBS -j oe
#
# Fase 8 -- la taglia del problema, a risorse fisse.
#
# Domanda: il costo per cella e' costante, o cambia con la griglia?
#
# Non e' una domanda oziosa. Il tempo per cella-passo (`per cell-step' nelle
# statistiche) dovrebbe essere piatto se il lavoro fosse proporzionale alle
# celle. Non lo e', per tre motivi che si distinguono solo guardando la curva:
#
#   in basso   griglie piccole: il transitorio della pipeline (P-1 stadi) e le
#              collettive di schur non sono ammortizzati, e i blocchi locali
#              stanno in cache. Il costo per cella e' dominato da cio' che non
#              dipende dalle celle.
#   in mezzo   la zona in cui il blocco locale esce dall'ultimo livello di
#              cache: il costo per cella sale e resta su.
#   in alto    la banda di memoria e' satura e il costo per cella si appiattisce
#              di nuovo, ma piu' in alto.
#
# Dove cade il gradino dice quale delle due configurazioni conviene per una
# griglia data, ed e' l'unica fase che risponde alla domanda pratica "per la
# taglia che devo simulare, cosa lancio".
#
# Le quattro configurazioni sono le estremita' vincenti della fase 5 piu' i due
# riferimenti seriali, che si fermano presto perche' a 320^3 un core solo
# costerebbe minuti per passo.
#
#   qsub scripts/study/08_size.sh

source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"

cd "${PBS_O_WORKDIR:-$STUDY_ROOT}"
study_begin 08_size
study_machine

SIZES="${SIZES:-64 96 128 160 192 224 256 288 320}"
SERIAL_MAX="${SERIAL_MAX:-128}"
STEPS="${STEPS:-10}"

CASE_SIMD=1
CASE_MPI=1
CASE_STEPS="$STEPS"
CASE_REPEATS="${REPEATS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-3000}"

full="$STUDY_PHYSICAL"
shape_full="$(study_auto_shape "$full")"
[[ -z "$shape_full" ]] && shape_full=""

for n in $SIZES; do
    echo "=== ${n}^3 ==="

    # Tutti i core in thread, dominio intero: nessun asse diviso, SIMD viva,
    # nessuna collettiva. E' l'estremita' buona della fase 5.
    for backend in schur pipeline; do
        study_case label="$backend 1x$full ${n}" backend="$backend" \
            omp=1 ranks=1 threads="$full" shape="1 1 1" grid="$n $n $n"
    done

    # Tutti i core in processi: nessun team di thread da aprire, ma il dominio
    # e' diviso in tre direzioni. E' l'altra estremita' buona.
    for backend in schur pipeline; do
        study_case label="$backend ${full}x1 ${n}" backend="$backend" \
            omp=0 ranks="$full" threads=1 shape="$shape_full" \
            grid="$n $n $n"
    done

    # Il riferimento seriale, finche' e' sostenibile.
    if [[ "$n" -le "$SERIAL_MAX" ]]; then
        for backend in schur pipeline; do
            study_case label="$backend seriale ${n}" backend="$backend" \
                omp=0 mpi=0 ranks=1 threads=1 grid="$n $n $n"
        done
    fi
    echo
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo "=== risultato ==="
    awk -F, -v phase=08_size '
    NR == 1 || $1 != phase || $32 != "ok" { next }
    {
        conf = ($7 == 0 && $8 == 1 ? "seriale" : ($8 == 1 ? "1 x " $9 : $8 " x 1"))
        k = $3 "," conf "," $10
        wall[k] = $17; rss[k] = $29
        # celle globali per passo, in nanosecondi per cella
        percell[k] = $17 * 1e6 / ($10 * $11 * $12)
        if (!($10 in seen_n)) { ns[++nn] = $10; seen_n[$10] = 1 }
        if (!(conf in seen_c)) { cs[++nc] = conf; seen_c[conf] = 1 }
    }
    END {
        for (a = 1; a <= nn; a++) for (b = a + 1; b <= nn; b++)
            if (ns[a] + 0 > ns[b] + 0) { t = ns[a]; ns[a] = ns[b]; ns[b] = t }
        for (ci = 1; ci <= nc; ci++) {
            conf = cs[ci]
            printf "\n  configurazione %s\n", conf
            printf "  %-8s %24s %24s\n", "", "schur", "pipeline"
            printf "  %-8s %11s %11s %11s %11s\n",
                   "griglia", "ms/passo", "ns/cella", "ms/passo", "ns/cella"
            for (j = 1; j <= nn; j++) {
                n = ns[j]
                ka = "schur," conf "," n; kb = "pipeline," conf "," n
                if (!(ka in wall) && !(kb in wall)) continue
                printf "  %-8s", n "^3"
                if (ka in wall) printf " %11.1f %11.3f", wall[ka], percell[ka]
                else printf " %11s %11s", "-", "-"
                if (kb in wall) printf " %11.1f %11.3f", wall[kb], percell[kb]
                else printf " %11s %11s", "-", "-"
                printf "\n"
            }
        }
        print ""
        print "  `ns/cella\x27 e\x27 il tempo di un passo diviso per le celle globali."
        print "  Se fosse una costante il codice sarebbe perfettamente scalabile"
        print "  nella taglia: dove sale, sta cambiando il regime -- cache che"
        print "  finisce, banda che satura, transitorio che non e\x27 ammortizzato."
    }' "$STUDY_CSV"
fi

study_end
