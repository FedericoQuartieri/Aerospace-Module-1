#!/usr/bin/env bash

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
launcher="${MPIRUN:-mpirun}"
compiler="${MPICC:-mpicc}"
repeats="${REPEATS:-1}"
steps="${STEPS:-200}"
pipeline_batch_lines="${PIPELINE_BATCH_LINES:-64}"
read -r -a ranks <<< "${MPI_RANKS:-1 2 4 8}"
read -r -a grids <<< "${GRID_SIZES:-192}"
read -r -a simd_modes <<< "${SIMD_MODES:-${SIMD:-0 1}}"
grid_tag="$(IFS=-; echo "${grids[*]}")"
simd_tag="$(IFS=-; echo "${simd_modes[*]}")"
results="${RESULTS:-$root/build/mpi_scaling_${grid_tag}_simd${simd_tag}.csv}"

if [[ "${ranks[0]}" != 1 ]]; then
    echo "MPI_RANKS must start with 1 to define the strong-scaling baseline" >&2
    exit 1
fi

mkdir -p "$(dirname -- "$results")"
printf 'grid,simd,ranks,time_s,speedup,efficiency\n' > "$results"

elapsed_from_output()
{
    awk '
        /Grid:/          { cells = $2 * $4 * $6 }
        /Time steps:/    { steps = $3 }
        /per cell-step:/ { cost = $3 }
        END {
            if (!cells || !steps || !cost) exit 1
            printf "%.9g", cost * 1e-8 * cells * steps
        }
    '
}

for grid in "${grids[@]}"; do
    for simd in "${simd_modes[@]}"; do
        if [[ "$simd" != 0 && "$simd" != 1 ]]; then
            echo "SIMD must be 0 or 1 (received: $simd)" >&2
            exit 1
        fi

        executable="$root/build/mpi_solver_${grid}_simd${simd}"
        compile_flags=(-std=c11 -O3 -I"$root/include"
            -DWIDTH="$grid" -DHEIGHT="$grid" -DDEPTH="$grid"
            -DSTEPS="$steps"
            -DPIPELINE_BATCH_LINES="$pipeline_batch_lines")
        if [[ "$simd" == 1 ]]; then
            compile_flags+=(-DUSE_SIMD)
            case "$(uname -m)" in
                x86_64|amd64) compile_flags+=(-mavx2) ;;
            esac
        fi
        "$compiler" "${compile_flags[@]}" "$root"/src/*.c \
            -o "$executable" -lm

        baseline=""
        for np in "${ranks[@]}"; do
            total="0"
            for ((run = 1; run <= repeats; run++)); do
                echo "Grid=${grid}^3, SIMD=$simd, MPI ranks=$np, run=$run/$repeats" >&2
                output="$($launcher -np "$np" "$executable")"
                elapsed="$(elapsed_from_output <<< "$output")"
                total="$(awk -v a="$total" -v b="$elapsed" 'BEGIN {print a + b}')"
            done

            average="$(awk -v t="$total" -v n="$repeats" 'BEGIN {print t / n}')"
            [[ -n "$baseline" ]] || baseline="$average"
            speedup="$(awk -v t1="$baseline" -v tp="$average" 'BEGIN {print t1 / tp}')"
            efficiency="$(awk -v s="$speedup" -v p="$np" 'BEGIN {print s / p}')"

            printf '%s,%s,%s,%.6f,%.6f,%.6f\n' \
                "$grid" "$simd" "$np" "$average" "$speedup" "$efficiency" \
                | tee -a "$results"
        done
    done
done

echo "Results written to $results"
