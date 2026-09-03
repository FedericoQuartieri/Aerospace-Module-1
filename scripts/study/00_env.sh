#!/usr/bin/env bash
#PBS -N nsb-00-env
#PBS -q cpu
#PBS -l select=1:ncpus=28
#PBS -l walltime=02:00:00
#PBS -j oe
#
# Fase 0 -- la macchina, i binari, e la sola cosa che rende sensato tutto il
# resto: che le configurazioni che stiamo per cronometrare risolvano lo stesso
# problema.
#
# Domanda: le 33 varianti di questo solutore (due backend, con e senza SIMD,
# con e senza thread, da 1 a 8 processi, tre valori di batch) danno la stessa
# risposta cifra per cifra?
#
# Perche' e' la prima fase e non un dettaglio: se due backend divergono, il
# confronto dei loro tempi non e' un confronto, e' un aneddoto su due
# programmi diversi.  Il repo dichiara questa proprieta' (README, "Both give
# the same answer, digit for digit"); qui la si verifica sulla macchina e con
# i flag con cui si misurera', perche' e' li' che potrebbe non valere.
#
# Costa poco: 64^3, 5 passi, qualche secondo a caso.  Compila anche tutte le
# varianti che le fasi successive useranno, cosi' un errore di compilazione
# esce adesso e non a meta' della notte.
#
# Coda `cpu' e non `scalability', al contrario di tutte le fasi che misurano.
# Qui non si cronometra niente: si confrontano cifre, e le cifre non cambiano
# se il nodo ha altri job addosso o se i processi sono in sovrannumero rispetto
# alle CPU concesse. In cambio si prende il walltime lungo, e le fasi che
# misurano non restano in coda dietro a questa per il nodo esclusivo.
#
#   qsub scripts/study/00_env.sh
#   ./scripts/study/00_env.sh          in locale, per provare

source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"

cd "${PBS_O_WORKDIR:-$STUDY_ROOT}"

# La 00 non si ri-sottomette da sola: le altre fasi dipendono dalla sua
# riuscita con -W depend=afterok, e devono partire quando ha finito davvero.
# Se il budget non le basta, il messaggio in fondo dice di rilanciarla.
STUDY_CHAINABLE=0
STUDY_BUDGET="${STUDY_BUDGET:-6000}"

study_begin 00_env
study_machine

# ------------------------------------------------------------------ ambiente

echo "=== ambiente ==="
printf 'compilatore:   %s\n' "$(${MPICC:-mpicc} --version 2>&1 | head -1)"
printf 'git:           %s\n' "$(git -C "$STUDY_ROOT" describe --always --dirty 2>/dev/null || echo n/d)"
printf 'branch:        %s\n' "$(git -C "$STUDY_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/d)"

# I componenti di lancio disponibili decidono se la fase 09 (multi-nodo) ha
# qualche speranza: senza `plm tm` PBS non sa avviare processi sugli altri
# nodi, ed e' esattamente il muro contro cui si e' fermata l'analisi
# precedente (MULTITHREAD.md §9.3).
if command -v ompi_info > /dev/null; then
    printf 'plm:           %s\n' \
        "$(ompi_info --param plm all 2>/dev/null | awk -F: '/MCA plm:/ {print $3}' | tr -d ' ' | sort -u | paste -sd, -)"
    printf 'btl:           %s\n' \
        "$(ompi_info --param btl all 2>/dev/null | awk -F: '/MCA btl:/ {print $3}' | tr -d ' ' | sort -u | paste -sd, -)"
fi
echo

# --------------------------------------------------------------- compilazione

echo "=== compilazione delle varianti ==="
build_failures=0
verdict=0
for backend in schur pipeline; do
    for simd in 0 1; do
        for omp in 0 1; do
            for mpi in 0 1; do
                printf '  %-9s simd=%s omp=%s mpi=%s  ' \
                    "$backend" "$simd" "$omp" "$mpi"
                if study_build "$backend" "$simd" "$omp" "$mpi" 64 > /dev/null; then
                    echo ok
                else
                    echo FALLITA
                    build_failures=$(( build_failures + 1 ))
                fi
            done
        done
    done
done

# I batch della pipeline che la fase 06 spazzola: compilarli qui li mette in
# cache una volta per tutte le fasi.
for batch in 8 16 32 128 256 512 1024; do
    printf '  pipeline  batch=%-4s              ' "$batch"
    if study_build pipeline 1 1 1 "$batch" > /dev/null; then
        echo ok
    else
        echo FALLITA
        build_failures=$(( build_failures + 1 ))
    fi
done
echo

if [[ "$build_failures" -gt 0 ]]; then
    echo "  $build_failures varianti non compilano: le fasi successive"
    echo "  lascerebbero righe con status=build. Meglio fermarsi qui."
    echo
    verdict=2
fi

# ----------------------------------------------------------------- coerenza

# Griglia piccola: qui non si misura niente, si confrontano cifre.
CASE_GRID="${GRID:-64 64 64}"
CASE_STEPS="${STEPS:-5}"
CASE_REPEATS=1
CASE_NORMS=1
CASE_TIMEOUT="${CASE_TIMEOUT:-900}"
RANKS="${RANKS:-1 2 4 8}"
THREADS="${THREADS:-1 4}"
SIMD_MODES="${SIMD_MODES:-0 1}"
SHAPES="${SHAPES:-8-1-1 1-8-1 1-1-8 2-2-2}"

echo "=== stessa risposta? (${CASE_GRID// /x}, $CASE_STEPS passi, norme attive) ==="

# Il riferimento: seriale puro, niente MPI, niente thread, niente SIMD.
study_case label="riferimento seriale" backend=schur \
    mpi=0 omp=0 simd=0 ranks=1 threads=1

for backend in schur pipeline; do
    for simd in $SIMD_MODES; do
        for ranks in $RANKS; do
            for threads in $THREADS; do
                shape="$(study_auto_shape "$ranks")"
                study_case label="$backend simd=$simd ${ranks}x${threads}" \
                    backend="$backend" simd="$simd" omp=1 mpi=1 \
                    ranks="$ranks" threads="$threads" shape="$shape"
            done
        done
    done
done

# La forma non deve cambiare il risultato piu' del numero di processi: e' la
# stessa proprieta', chiesta all'asse che di solito non viene diviso.
for spec in $SHAPES; do
    shape="${spec//-/ }"
    ranks=$(( ${spec%%-*} * $(cut -d- -f2 <<< "$spec") * ${spec##*-} ))
    for backend in schur pipeline; do
        study_case label="$backend forma ${spec//-/x}" backend="$backend" \
            ranks="$ranks" threads=1 shape="$shape"
    done
done

# Il batch e' una scelta di come si spedisce, non di cosa si calcola.
for batch in ${BATCHES:-8 64 1024}; do
    study_case label="pipeline batch=$batch" backend=pipeline batch="$batch" \
        ranks="${BATCH_RANKS:-4}" threads="${BATCH_THREADS:-2}" \
        shape="$(study_auto_shape "${BATCH_RANKS:-4}")"
done

# ---------------------------------------------------------------- il verdetto

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== verdetto ==="
    awk -F, -v phase=00_env '
    NR == 1 { next }
    $1 != phase || $32 != "ok" { next }
    {
        rows++
        key = $30 "|" $31            # L2 u_x, L2 p
        if (!(key in seen)) { seen[key] = $2; distinct[++n] = key }
        count[key]++
    }
    END {
        if (rows == 0) { print "  nessuna riga da confrontare"; exit }
        printf "  %d configurazioni confrontate\n", rows
        if (n == 1) {
            printf "  tutte danno le stesse norme, cifra per cifra:\n"
            split(distinct[1], p, "|")
            printf "    L2 u_x = %s\n    L2 p   = %s\n\n", p[1], p[2]
            print  "  I tempi delle fasi successive confrontano lo stesso calcolo."
        } else {
            printf "\n  ATTENZIONE: %d risposte diverse.\n\n", n
            for (i = 1; i <= n; i++) {
                split(distinct[i], p, "|")
                printf "    %-28s L2 u_x = %s  L2 p = %s  (%d configurazioni)\n",
                       seen[distinct[i]], p[1], p[2], count[distinct[i]]
            }
            printf "\n  Finche\x27 questo non torna, i tempi delle altre fasi\n"
            printf "  confrontano programmi diversi. Non proseguire.\n"
            exit 3
        }
    }' "$STUDY_CSV" || verdict=$?
fi

study_end

# L'uscita non e' un dettaglio: le altre fasi sono sottomesse con
# `-W depend=afterok', quindi PBS le lascia partire solo se questa finisce
# bene. Un verdetto negativo che uscisse con 0 farebbe partire lo studio sopra
# a un solutore che non da' la stessa risposta, ed e' esattamente il caso in
# cui non deve partire.
if [[ "$verdict" -ne 0 ]]; then
    echo
    echo "esco con $verdict: le fasi che dipendono da questa non partiranno."
fi
exit "$verdict"
