#!/usr/bin/env bash

set -euo pipefail

# I numeri passano per printf '%.6f' e per awk: con un locale che usa la
# virgola decimale bash rifiuta "0.065964" come numero.
export LC_ALL=C

# Strong scaling del solutore su un numero crescente di processi.
#
# Il solutore non e' piu' il glob piatto src/*.c: il backend tridiagonale sta
# in src/tridiag/$(TRIDIAG)/ e i flag li conosce solo il Makefile, quindi si
# compila attraverso di lui, come fa scripts/study/lib.sh.
#
#   ./scripts/run_mpi_scaling.sh
#   TRIDIAG=pipeline PIPELINE_BATCH_LINES=128 ./scripts/run_mpi_scaling.sh
#   GRID_SIZES="64 128" MPI_RANKS="1 2 4" SIMD_MODES=1 ./scripts/run_mpi_scaling.sh

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
launcher="${MPIRUN:-mpirun}"
repeats="${REPEATS:-1}"
steps="${STEPS:-200}"
tridiag="${TRIDIAG:-schur}"
pipeline_batch_lines="${PIPELINE_BATCH_LINES:-64}"
build_log="$root/build/mpi_scaling.build.log"
read -r -a ranks <<< "${MPI_RANKS:-1 2 4 8}"
read -r -a grids <<< "${GRID_SIZES:-192}"
read -r -a simd_modes <<< "${SIMD_MODES:-${SIMD:-0 1}}"
grid_tag="$(IFS=-; echo "${grids[*]}")"
simd_tag="$(IFS=-; echo "${simd_modes[*]}")"
results="${RESULTS:-$root/build/mpi_scaling_${tridiag}_${grid_tag}_simd${simd_tag}.csv}"

if [[ "${ranks[0]}" != 1 ]]; then
    echo "MPI_RANKS must start with 1 to define the strong-scaling baseline" >&2
    exit 1
fi

mkdir -p "$(dirname -- "$results")"
printf 'grid,simd,ranks,time_s,speedup,efficiency,backend\n' > "$results"

# Il tempo totale e' il passo del rank piu' lento per il numero di passi.
# "per cell-step" non va piu' bene: print_stats lo riferisce alle celle del
# blocco locale, non alla griglia globale, e moltiplicarlo per quella
# gonfierebbe il tempo di un fattore pari ai processi.
elapsed_from_output()
{
    awk '
        /Time steps:/    { steps = $3 }
        /wall per step:/ { wall_ms = $4 }
        END {
            if (!steps || !wall_ms) exit 1
            printf "%.9g", wall_ms * 1e-3 * steps
        }
    '
}

for grid in "${grids[@]}"; do
    for simd in "${simd_modes[@]}"; do
        if [[ "$simd" != 0 && "$simd" != 1 ]]; then
            echo "SIMD must be 0 or 1 (received: $simd)" >&2
            exit 1
        fi

        executable="$root/build/mpi_solver_${tridiag}_${grid}_simd${simd}"
        # La griglia e' una costante di compilazione, come per i test: cosi'
        # il binario si lancia senza file di configurazione.
        if ! make -s -B --no-print-directory -C "$root" \
                TRIDIAG="$tridiag" MPI=1 OMP=0 SIMD="$simd" \
                PIPELINE_BATCH_LINES="$pipeline_batch_lines" \
                EXTRA_CPPFLAGS="-DDEFAULT_WIDTH=$grid -DDEFAULT_HEIGHT=$grid \
                    -DDEFAULT_DEPTH=$grid -DDEFAULT_STEPS=$steps" \
                solver > "$build_log" 2>&1; then
            echo "compilazione fallita, vedi $build_log" >&2
            sed 's/^/    /' "$build_log" >&2
            exit 1
        fi
        mv "$root/solver" "$executable"

        baseline=""
        for np in "${ranks[@]}"; do
            total="0"
            for ((run = 1; run <= repeats; run++)); do
                echo "Backend=$tridiag, Grid=${grid}^3, SIMD=$simd, MPI ranks=$np, run=$run/$repeats" >&2
                output="$($launcher -np "$np" "$executable")"
                elapsed="$(elapsed_from_output <<< "$output")"
                total="$(awk -v a="$total" -v b="$elapsed" 'BEGIN {print a + b}')"
            done

            average="$(awk -v t="$total" -v n="$repeats" 'BEGIN {print t / n}')"
            [[ -n "$baseline" ]] || baseline="$average"
            speedup="$(awk -v t1="$baseline" -v tp="$average" 'BEGIN {print t1 / tp}')"
            efficiency="$(awk -v s="$speedup" -v p="$np" 'BEGIN {print s / p}')"

            printf '%s,%s,%s,%.6f,%.6f,%.6f,%s\n' \
                "$grid" "$simd" "$np" "$average" "$speedup" "$efficiency" \
                "$tridiag" | tee -a "$results"
        done
    done
done

echo "Results written to $results"
