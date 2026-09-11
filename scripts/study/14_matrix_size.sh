#!/usr/bin/env bash
#PBS -N nsb-14-size
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Fase 14 -- la taglia del problema: scaling forte, debole, e il muro.
#
# Tre domande che vogliono le stesse misure e che quindi conviene fare insieme.
#
#   forte    a problema fisso, quanto si accorcia il tempo aggiungendo unita'.
#            E' la domanda che fanno tutti e la meno onesta su uno stencil:
#            il blocco locale si rimpicciolisce finche' entra in cache e lo
#            speedup diventa superlineare per un motivo che non e' il
#            parallelismo.
#
#   debole   a lavoro per unita' fisso, il tempo resta costante mentre il
#            problema cresce. Su un codice limitato dalla banda e' la misura
#            che dice davvero se il parallelismo funziona.
#
#   il muro  il costo per cella in funzione della taglia, a parallelismo fisso.
#            Cresce quando il blocco locale esce dalla cache, e il punto in cui
#            cresce e' una proprieta' della macchina, non del codice: saperlo
#            e' quello che permette di leggere le altre due.
#
# La memoria di picco viaggia nel CSV (colonna rss_mb), e per la pipeline non
# e' un dettaglio: tiene c' e d' di tutto il blocco locale per tre componenti,
# quindi paga in memoria quello che Schur paga in aritmetica. Questa fase e'
# l'unico posto dove quel prezzo si vede su un asse.
#
#   qsub scripts/study/14_matrix_size.sh

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub va fatto dalla radice del repo; qui sono in $PWD" >&2
    exit 1
fi

source scripts/study/lib.sh
source scripts/study/matrix.sh

STUDY_CHAIN_MAX="$MATRIX_CHAIN_MAX"
study_begin 14_matrix_size
study_machine

SIZES="${SIZES:-32 48 64 96 128 160 192 224 256}"
# I piazzamenti su cui si percorre la scala delle taglie: seriale, tutta la
# macchina a thread, tutta la macchina a rank, e il misto.
SIZE_PLACEMENTS="${SIZE_PLACEMENTS:-1x1 1x56 56x1 8x7}"
# Scaling forte: la taglia resta, le unita' crescono.
STRONG_GRID="${STRONG_GRID:-224}"
STRONG_RANKS="${STRONG_RANKS:-1 2 4 7 8 14 28 56}"
# Scaling debole: celle per rank costanti. Ogni voce e' "rank : griglia".
WEAK_CONFIGS="${WEAK_CONFIGS:-1:64 64 64|2:128 64 64|4:128 128 64|8:128 128 128|28:224 224 128|56:224 224 224}"

CASE_OMP=1
CASE_MPI=1
CASE_REPEATS="${REPEATS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-1500}"

echo "=== il muro: costo per cella al crescere della taglia ==="
for place in $SIZE_PLACEMENTS; do
    r="${place%x*}"
    t="${place#*x}"
    [[ $(( r * t )) -le "$(matrix_units)" ]] || continue
    shape="$(study_auto_shape "$r")"
    [[ -n "$shape" ]] || continue

    for backend in $MATRIX_BACKENDS; do
        for simd in $MATRIX_SIMD; do
            for n in $SIZES; do
                matrix_shape_fits "$shape" "$n $n $n" || continue
                study_case label="$backend $place s$simd N=$n" \
                    backend="$backend" simd="$simd" ranks="$r" threads="$t" \
                    shape="$shape" grid="$n $n $n" \
                    steps="$(matrix_steps "$n")"
            done
        done
    done
done
echo

echo "=== scaling forte: ${STRONG_GRID}^3, unita' crescenti ==="
steps="$(matrix_steps "$STRONG_GRID")"
grid="$STRONG_GRID $STRONG_GRID $STRONG_GRID"
for backend in $MATRIX_BACKENDS; do
    for n in $STRONG_RANKS; do
        shape="$(study_auto_shape "$n")"
        [[ -n "$shape" ]] || continue
        matrix_shape_fits "$shape" "$grid" || continue
        # Due modi di spendere le stesse unita': tutti rank, oppure un rank
        # con altrettanti thread. La differenza e' il costo di dividere.
        study_case label="$backend forte R=$n T=1" backend="$backend" \
            ranks="$n" threads=1 shape="$shape" simd=1 \
            grid="$grid" steps="$steps"
        study_case label="$backend forte R=1 T=$n" backend="$backend" \
            ranks=1 threads="$n" shape="1 1 1" simd=1 \
            grid="$grid" steps="$steps"
    done
done
echo

echo "=== scaling debole: celle per rank costanti ==="
IFS='|' read -r -a weak <<< "$WEAK_CONFIGS"
for entry in "${weak[@]}"; do
    n="${entry%%:*}"
    grid="${entry#*:}"
    shape="$(study_auto_shape "$n")"
    [[ -n "$shape" ]] || continue
    matrix_shape_fits "$shape" "$grid" || continue
    read -r wx wy wz <<< "$grid"

    for backend in $MATRIX_BACKENDS; do
        for simd in $MATRIX_SIMD; do
            study_case label="$backend debole R=$n s$simd" \
                backend="$backend" simd="$simd" ranks="$n" threads=1 \
                shape="$shape" grid="$grid" \
                steps="$(matrix_steps "$wx")" \
                note="debole, ${wx}x${wy}x${wz}"
        done
    done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== costo per cella (1e-8 s) e memoria di picco, per taglia ==="
    awk -F, -v phase=14_matrix_size '
    NR == 1 || $1 != phase || $(NF - 1) != "ok" || $2 !~ / N=/ { next }
    {
        k = $3 "," $5 "," $8 "x" $9
        cell[k "," $10] = $28 + 0
        rss[k "," $10] = $29 + 0
        if (!(k in seen)) { keys[++nk] = k; seen[k] = 1 }
        if (!($10 in seen_n)) { ns[++nn] = $10 + 0; seen_n[$10] = 1 }
    }
    END {
        printf "  %-10s %-5s %-8s", "backend", "simd", "rankxthr"
        for (j = 1; j <= nn; j++) printf " %8s", ns[j]
        printf "\n"
        for (i = 1; i <= nk; i++) {
            split(keys[i], p, ",")
            printf "  %-10s %-5s %-8s", p[1], p[2], p[3]
            for (j = 1; j <= nn; j++) {
                kk = keys[i] "," ns[j]
                if (kk in cell) printf " %8.2f", cell[kk]
                else printf " %8s", "."
            }
            printf "\n"
        }
        print ""
        print "  Costo per cella per passo: se fosse solo calcolo resterebbe"
        print "  costante. Dove sale, il blocco locale e\x27 uscito dalla cache."
    }' "$STUDY_CSV"
fi

study_end
