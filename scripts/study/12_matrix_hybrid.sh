#!/usr/bin/env bash
#PBS -N nsb-12-hybrid
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Fase 12 -- il prodotto pieno rank x thread.
#
# La 05 misura la diagonale: rank x thread costante, uguale ai core fisici.
# Quella diagonale confonde tre cose -- quanti thread ha ogni rank, quanto e'
# carica la macchina, e come e' diviso il dominio -- e infatti la lettura di
# MULTITHREAD.md 9.2 ci si e' rotta sopra (vedi hybrid-collapse).
#
# Questa fase riempie il rettangolo invece della diagonale: OGNI coppia (rank,
# thread) che sta nel nodo, per entrambi i backend, con e senza SIMD. Con il
# rettangolo pieno le tre cose si separano da sole:
#
#   - a rank fissi, leggendo lungo i thread, cambiano solo i thread;
#   - a thread fissi, leggendo lungo i rank, cambia solo la divisione;
#   - le anti-diagonali a prodotto costante sono le righe della 05, e adesso
#     hanno intorno il contesto che mancava.
#
# La previsione, dal codice: sulla pipeline i thread rendono poco comunque,
# perche' ogni livello e' una barriera (vedi PIPELINE, batch e barriere), e
# quindi la colonna a molti thread dovrebbe essere piatta e migliorare con
# batch grandi -- che e' quello che misura la 13. Su Schur invece i thread
# rendono finche' l'asse non e' diviso.
#
#   qsub scripts/study/12_matrix_hybrid.sh

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub va fatto dalla radice del repo; qui sono in $PWD" >&2
    exit 1
fi

source scripts/study/lib.sh
source scripts/study/matrix.sh

STUDY_CHAIN_MAX="$MATRIX_CHAIN_MAX"
study_begin 12_matrix_hybrid
study_machine

# La taglia principale, dove si prova il rettangolo con e senza SIMD.
FULL_GRID="${FULL_GRID:-224}"
# Le altre, dove si prova solo con SIMD acceso.
PLAIN_GRIDS="${PLAIN_GRIDS:-128}"

CASE_OMP=1
CASE_MPI=1
CASE_REPEATS="${REPEATS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-1200}"

emit_rectangle()
{
    local grid="$1" simd="$2" steps="$3" backend r t shape

    for backend in $MATRIX_BACKENDS; do
        while read -r r t; do
            shape="$(study_auto_shape "$r")"
            [[ -n "$shape" ]] || continue
            matrix_shape_fits "$shape" "$grid" || continue
            study_case label="$backend R=$r T=$t s$simd" \
                backend="$backend" ranks="$r" threads="$t" shape="$shape" \
                simd="$simd" grid="$grid" steps="$steps"
        done < <(matrix_pairs)
    done
}

steps="$(matrix_steps "$FULL_GRID")"
for simd in $MATRIX_SIMD; do
    echo "=== ${FULL_GRID}^3, simd=$simd: ogni coppia rank x thread ==="
    emit_rectangle "$FULL_GRID $FULL_GRID $FULL_GRID" "$simd" "$steps"
    echo
done

for m in $PLAIN_GRIDS; do
    echo "=== ${m}^3, simd=1: lo stesso rettangolo su un'altra taglia ==="
    emit_rectangle "$m $m $m" 1 "$(matrix_steps "$m")"
    echo
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== il rettangolo: ms/passo, righe rank, colonne thread ==="
    awk -F, -v phase=12_matrix_hybrid -v tlist="$MATRIX_THREADS" '
    NR == 1 || $1 != phase || $(NF - 1) != "ok" { next }
    {
        k = $3 "," $5 "," $10
        wall[k "," $8 "," $9] = $17 + 0
        if (!(k in seen)) { keys[++nk] = k; seen[k] = 1 }
        if (!($8 in seen_r)) { rs[++nr] = $8 + 0; seen_r[$8] = 1 }
    }
    END {
        nt = split(tlist, ts, " ")
        for (i = 1; i <= nk; i++) {
            split(keys[i], p, ",")
            printf "\n  %s  simd=%s  %s^3\n", p[1], p[2], p[3]
            printf "  %-6s", "rank"
            for (j = 1; j <= nt; j++) printf " %8s", "T=" ts[j]
            printf "\n"
            for (r = 1; r <= nr; r++) {
                printf "  %-6s", rs[r]
                for (j = 1; j <= nt; j++) {
                    kk = keys[i] "," rs[r] "," ts[j]
                    if (kk in wall) printf " %8.1f", wall[kk]
                    else printf " %8s", "."
                }
                printf "\n"
            }
        }
        print ""
        print "  Un punto e\x27 una coppia che non ci sta nel nodo. Le anti-diagonali"
        print "  a prodotto costante sono le righe della fase 05."
    }' "$STUDY_CSV"
fi

study_end
