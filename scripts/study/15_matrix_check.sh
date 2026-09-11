#!/usr/bin/env bash
#PBS -N nsb-15-check
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Fase 15 -- la matrice risolve ancora il problema giusto?
#
# Le altre cinque fasi misurano tempi. Un tempo di una configurazione che da'
# la risposta sbagliata non e' un tempo lento, e' un numero senza significato,
# e la campagna ne produce migliaia senza guardare nemmeno una volta il
# risultato. Questa fase guarda solo quello.
#
# Ogni caso gira con BENCH_NORMS=1, che calcola le norme dell'errore contro la
# soluzione esatta, e le mette nel CSV (colonne l2_ux e l2_p). La proprieta' da
# verificare e' che siano LE STESSE per ogni configurazione: il risultato non
# deve dipendere da quanti processi, quanti thread, quale backend, quale forma
# della griglia, quale batch. E' la stessa proprieta' che check_pipeline.sh
# verifica in piccolo, qui allargata a tutta la matrice.
#
# Griglia piccola apposta: la correttezza non ha bisogno della taglia, e a
# 64^3 questi casi costano pochi secondi l'uno. La colonna che conta e' la
# differenza fra le norme, non il loro valore.
#
# Un caso che qui esce diverso dagli altri invalida tutte le righe delle altre
# fasi con la stessa configurazione, e va risolto prima di leggere qualunque
# tempo.
#
#   qsub scripts/study/15_matrix_check.sh

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub va fatto dalla radice del repo; qui sono in $PWD" >&2
    exit 1
fi

source scripts/study/lib.sh
source scripts/study/matrix.sh

STUDY_CHAIN_MAX="$MATRIX_CHAIN_MAX"
study_begin 15_matrix_check
study_machine

GRID="${CHECK_GRID:-64}"
CHECK_RANKS="${CHECK_RANKS:-1 2 4 7 8 14 28}"
CHECK_THREADS="${CHECK_THREADS:-1 2 7 8}"
CHECK_BATCHES="${CHECK_BATCHES:-1 7 64 1024}"

grid="$GRID $GRID $GRID"
steps="$(matrix_steps "$GRID")"

CASE_MPI=1
CASE_OMP=1
# Una ripetizione basta: qui non si cronometra niente.
CASE_REPEATS=1
CASE_NORMS=1
CASE_TIMEOUT="${CASE_TIMEOUT:-600}"

echo "=== ${GRID}^3: ogni forma della griglia di processi, un thread ==="
for backend in $MATRIX_BACKENDS; do
    for simd in $MATRIX_SIMD; do
        for n in $CHECK_RANKS; do
            while read -r shape; do
                matrix_shape_fits "$shape" "$grid" || continue
                study_case label="$backend R=$n ${shape// /x} s$simd" \
                    backend="$backend" simd="$simd" ranks="$n" threads=1 \
                    shape="$shape" grid="$grid" steps="$steps"
            done < <(matrix_shapes "$n")
        done
    done
done
echo

echo "=== ${GRID}^3: i thread non devono cambiare il risultato ==="
for backend in $MATRIX_BACKENDS; do
    for simd in $MATRIX_SIMD; do
        for t in $CHECK_THREADS; do
            for n in 1 4 8; do
                [[ $(( n * t )) -le "$(matrix_units)" ]] || continue
                shape="$(study_auto_shape "$n")"
                [[ -n "$shape" ]] || continue
                study_case label="$backend R=$n T=$t s$simd" \
                    backend="$backend" simd="$simd" ranks="$n" threads="$t" \
                    shape="$shape" grid="$grid" steps="$steps"
            done
        done
    done
done
echo

echo "=== ${GRID}^3: nemmeno il batch della pipeline deve cambiarlo ==="
for simd in $MATRIX_SIMD; do
    for b in $CHECK_BATCHES; do
        for n in 1 8; do
            shape="$(study_auto_shape "$n")"
            study_case label="pipeline R=$n b=$b s$simd" \
                backend=pipeline batch="$b" simd="$simd" ranks="$n" \
                threads=1 shape="$shape" grid="$grid" steps="$steps"
        done
    done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== verdetto: quanto si discostano le norme fra tutte le configurazioni ==="
    awk -F, -v phase=15_matrix_check '
    function abs(x) { return x < 0 ? -x : x }
    NR == 1 || $1 != phase || $33 != "ok" || $30 == "" { next }
    {
        n++
        ux[n] = $30 + 0; p[n] = $31 + 0; who[n] = $2
        # Il riferimento e\x27 il primo caso letto, qualunque sia: la proprieta\x27
        # da verificare e\x27 che tutti diano lo stesso, non che diano un valore
        # deciso in anticipo.
        if (n == 1) { rux = ux[1]; rp = p[1]; rwho = who[1] }
    }
    END {
        if (n == 0) { print "  nessuna norma nel CSV"; exit }
        worst_ux = 0; worst_p = 0
        worst_ux_who = rwho; worst_p_who = rwho
        for (i = 1; i <= n; i++) {
            du = (rux != 0) ? abs(ux[i] - rux) / abs(rux) : abs(ux[i])
            dp = (rp  != 0) ? abs(p[i]  - rp)  / abs(rp)  : abs(p[i])
            if (du > worst_ux) { worst_ux = du; worst_ux_who = who[i] }
            if (dp > worst_p)  { worst_p  = dp; worst_p_who  = who[i] }
        }
        printf "  configurazioni confrontate: %d\n", n
        printf "  riferimento:                %s\n", rwho
        printf "  scarto relativo massimo su |u_x|:  %.3e   (%s)\n",
               worst_ux, worst_ux_who
        printf "  scarto relativo massimo su |p|:    %.3e   (%s)\n",
               worst_p, worst_p_who
        print  ""
        if (worst_ux < 1e-10 && worst_p < 1e-10) {
            print "  Tutte le configurazioni danno la stessa risposta: i tempi"
            print "  delle altre fasi confrontano la stessa cosa."
        } else {
            print "  ATTENZIONE: qualche configurazione risolve un problema diverso."
            print "  I tempi che la riguardano non vanno letti finche\x27 non si sa"
            print "  perche\x27. In double lo scarto atteso e\x27 zero, non `piccolo\x27."
        }
    }' "$STUDY_CSV"
fi

study_end
