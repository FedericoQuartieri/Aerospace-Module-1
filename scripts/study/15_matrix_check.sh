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
    echo "qsub must be run from the repo root; I am in $PWD" >&2
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

# Il riferimento vero di tutta la fase. Qui non serve solo a normalizzare i
# tempi: e' la norma L2 contro cui tutte le righe seguenti devono combaciare.
# Un binario senza MPI e senza OpenMP non puo' sbagliare per colpa di una
# divisione o di un thread, quindi se una forma se ne discosta e' quella forma
# ad avere torto, non il riferimento.
echo "=== the baselines: serial and single process ==="
for backend in $MATRIX_BACKENDS; do
    for simd in $MATRIX_SIMD; do
        study_baseline label="$backend s$simd" \
            backend="$backend" simd="$simd" grid="$grid" steps="$steps"
    done
done
for simd in $MATRIX_SIMD; do
    for b in $CHECK_BATCHES; do
        study_baseline label="pipeline b=$b s$simd" \
            backend=pipeline batch="$b" simd="$simd" \
            grid="$grid" steps="$steps"
    done
done
echo

echo "=== ${GRID}^3: every process-grid shape, one thread ==="
for backend in $MATRIX_BACKENDS; do
    for simd in $MATRIX_SIMD; do
        for n in $CHECK_RANKS; do
            forme=()
            mapfile -t forme < <(matrix_shapes "$n")
            for shape in "${forme[@]}"; do
                matrix_shape_fits "$shape" "$grid" || continue
                study_case label="$backend R=$n ${shape// /x} s$simd" \
                    backend="$backend" simd="$simd" ranks="$n" threads=1 \
                    shape="$shape" grid="$grid" steps="$steps"
            done
        done
    done
done
echo

echo "=== ${GRID}^3: threads must not change the result ==="
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

echo "=== ${GRID}^3: nor must the pipeline batch change it ==="
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
    echo "=== verdict: how far the norms drift across all configurations ==="
    awk -F, -v phase=15_matrix_check '
    function abs(x) { return x < 0 ? -x : x }
    NR == 1 || $1 != phase || $(NF - 1) != "ok" || $30 == "" { next }
    {
        n++
        ux[n] = $30 + 0; p[n] = $31 + 0; who[n] = $2
        # Il riferimento e\x27 il primo caso letto, qualunque sia: la proprieta\x27
        # da verificare e\x27 che tutti diano lo stesso, non che diano un valore
        # deciso in anticipo.
        if (n == 1) { rux = ux[1]; rp = p[1]; rwho = who[1] }
    }
    END {
        if (n == 0) { print "  no norms in the CSV"; exit }
        worst_ux = 0; worst_p = 0
        worst_ux_who = rwho; worst_p_who = rwho
        for (i = 1; i <= n; i++) {
            du = (rux != 0) ? abs(ux[i] - rux) / abs(rux) : abs(ux[i])
            dp = (rp  != 0) ? abs(p[i]  - rp)  / abs(rp)  : abs(p[i])
            if (du > worst_ux) { worst_ux = du; worst_ux_who = who[i] }
            if (dp > worst_p)  { worst_p  = dp; worst_p_who  = who[i] }
        }
        printf "  configurations compared: %d\n", n
        printf "  reference:               %s\n", rwho
        printf "  largest relative drift on |u_x|:  %.3e   (%s)\n",
               worst_ux, worst_ux_who
        printf "  largest relative drift on |p|:    %.3e   (%s)\n",
               worst_p, worst_p_who
        print  ""
        if (worst_ux < 1e-10 && worst_p < 1e-10) {
            print "  Every configuration gives the same answer: the timings of"
            print "  the other phases compare the same thing."
        } else {
            print "  WARNING: some configuration is solving a different problem."
            print "  Its timings must not be read until the reason is known."
            print "  In double the expected drift is zero, not `small\x27."
        }
    }' "$STUDY_CSV"
fi

study_end
