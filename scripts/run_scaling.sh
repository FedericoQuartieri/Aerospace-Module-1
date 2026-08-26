#!/usr/bin/env bash
#
# Misura di quanto il calcolo parallelo conviene.
#
# Due studi, che rispondono a due domande diverse:
#
#   strong  a parita' di problema, quanto si accorcia il tempo aggiungendo
#           processi? Ogni processo riceve una fetta piu' piccola.
#
#   weak    a parita' di lavoro per processo, il tempo resta lo stesso mentre
#           il problema cresce? Per un codice a stencil come questo e' la
#           misura piu' onesta: il limite vero e' la banda di memoria, e lo
#           strong scaling la satura in fretta su una macchina sola.
#
#   hybrid  a parita' di core occupati, conviene di piu' spenderli in processi
#           o in thread? La domanda non e' oziosa: i processi dividono il
#           dominio, e appena una direzione e' divisa i kernel SIMD spariscono
#           e il complemento di Schur passa da una risoluzione locale a tre.
#           I thread dividono le linee, che sono indipendenti comunque, quindi
#           non pagano ne' l'una ne' l'altra cosa.
#
# Ogni riga contiene il numero di processi, la forma della griglia di
# processi, la griglia globale, il tempo per passo e la quota passata dentro
# MPI. I risultati finiscono in build/scaling/results.csv.

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$root/build/scaling"
results="$build_dir/results${RESULTS_SUFFIX:-}.csv"
executable="$build_dir/paper_man"
compiler="${MPICC:-mpicc}"
steps="${STEPS:-20}"
# Su un portatile il rumore di fondo e' molto: ogni caso si ripete e si tiene
# il tempo migliore, che e' quello meno contaminato da altre attivita'.
repeats="${REPEATS:-3}"
# SIMD=0 disattiva i kernel vettorizzati: serve a confrontare alla pari, dato
# che quelli valgono solo sulle direzioni non divise.
simd_flags="-mavx2 -DUSE_SIMD"
[[ "${SIMD:-1}" == "0" ]] && simd_flags=""

# processi : forma della griglia di processi : griglia globale
strong_configs=(
    "1 : 1 1 1 : 128 128 128"
    "2 : 1 1 2 : 128 128 128"
    "4 : 1 2 2 : 128 128 128"
    "8 : 2 2 2 : 128 128 128"
)

# 64^3 celle per processo in tutti i casi
weak_configs=(
    "1 : 1 1 1 : 64 64 64"
    "2 : 1 1 2 : 64 64 128"
    "4 : 1 2 2 : 64 128 128"
    "8 : 2 2 2 : 128 128 128"
)

# Stessa griglia e stesso numero di core, spartiti diversamente fra processi e
# thread. Le righe con un processo solo tengono ogni direzione intera: la SIMD
# resta viva e Schur non ha interfacce da ricucire.
hybrid_configs=(
    "1 x 1 : 1 1 1 : 128 128 128"
    "1 x 2 : 1 1 1 : 128 128 128"
    "2 x 1 : 1 1 2 : 128 128 128"
    "1 x 4 : 1 1 1 : 128 128 128"
    "2 x 2 : 1 1 2 : 128 128 128"
    "4 x 1 : 1 2 2 : 128 128 128"
    "1 x 8 : 1 1 1 : 128 128 128"
    "2 x 4 : 1 1 2 : 128 128 128"
    "4 x 2 : 1 2 2 : 128 128 128"
    "8 x 1 : 2 2 2 : 128 128 128"
)

core_sources=()
for source in "$root"/src/*.c; do
    [[ "$(basename -- "$source")" == "main.c" ]] || core_sources+=("$source")
done

mkdir -p "$build_dir"
trap 'rm -f "$executable"' EXIT
printf '%s\n' 'study,procs,threads,px,py,pz,nx,ny,nz,steps,wall_ms,mpi_ms' \
    > "$results"

run_case()
{
    local study="$1" procs="$2" threads="$3" shape="$4" grid="$5"
    read -r px py pz <<< "$shape"
    read -r nx ny nz <<< "$grid"

    printf 'Running %-6s procs=%-2s threads=%-2s shape=%sx%sx%s grid=%sx%sx%s\n' \
        "$study" "$procs" "$threads" "$px" "$py" "$pz" "$nx" "$ny" "$nz"

    # Il ramo a thread si compila solo quando serve: la build OpenMP costa un
    # punto percentuale anche a un thread solo, e le due misure vanno tenute
    # separate.
    local omp_flags=""
    [[ "$threads" -gt 1 ]] && omp_flags="-fopenmp -DUSE_OMP"

    "$compiler" -std=gnu11 -O3 -Wall -Wextra -I"$root/include" \
        $simd_flags $omp_flags -DUSE_MPI \
        -DDEFAULT_WIDTH="$nx" -DDEFAULT_HEIGHT="$ny" -DDEFAULT_DEPTH="$nz" \
        -DDEFAULT_T=1e-1 -DDEFAULT_STEPS="$steps" \
        "$root/test/paper_man.c" "${core_sources[@]}" -lm -o "$executable"

    # --bind-to none e' obbligatorio, non un dettaglio: per default mpirun
    # inchioda ogni processo a un core solo, e i thread di quel processo se lo
    # spartiscono invece di prendersene uno a testa.  Senza questa opzione la
    # colonna dei thread misura zero guadagno e la misura non dice niente.
    local best_wall="" best_mpi="" output wall mpi
    for ((run = 0; run < repeats; run++)); do
        output="$(OMP_NUM_THREADS="$threads" \
                  mpirun --oversubscribe --bind-to none \
                  -n "$procs" "$executable" \
                  "$px" "$py" "$pz")"
        wall="$(awk '/wall per step:/ {print $4}' <<< "$output")"
        mpi="$(awk '/mpi per step:/  {print $4}' <<< "$output")"

        if [[ -z "$best_wall" ]] || \
           awk "BEGIN { exit !($wall < $best_wall) }"; then
            best_wall="$wall"
            best_mpi="$mpi"
        fi
    done
    printf '    migliore di %s: %s ms\n' "$repeats" "$best_wall"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$study" "$procs" "$threads" "$px" "$py" "$pz" \
        "$nx" "$ny" "$nz" "$steps" "$best_wall" "$best_mpi" >> "$results"
}

for study in strong weak; do
    declare -n configs="${study}_configs"
    for config in "${configs[@]}"; do
        IFS=':' read -r procs shape grid <<< "$config"
        run_case "$study" "${procs// /}" 1 "$(echo "$shape")" "$(echo "$grid")"
    done
done

# Lo studio ibrido si salta con HYBRID=0: senza OpenMP nel compilatore le sue
# righe non si possono compilare.
if [[ "${HYBRID:-1}" != "0" ]]; then
    for config in "${hybrid_configs[@]}"; do
        IFS=':' read -r spec shape grid <<< "$config"
        read -r procs _ threads <<< "$spec"
        run_case hybrid "$procs" "$threads" \
            "$(echo "$shape")" "$(echo "$grid")"
    done
fi

printf '\nRisultati in %s\n\n' "$results"

# Nello strong scaling il tempo dovrebbe dimezzarsi raddoppiando i processi,
# quindi l efficienza e (t1/tP)/P. Nel weak il lavoro per processo non cambia,
# quindi il tempo dovrebbe restare costante e l efficienza e t1/tP.
awk -F, '
NR == 1 { next }
{
    if ($1 != study) {
        study = $1
        base = $11
        printf "\n%s scaling\n", study
        printf "  %-9s %-9s %-13s %11s %11s %9s %9s\n",
               "proc x th", "forma", "griglia", "wall/passo", "mpi/passo",
               (study == "weak" ? "t1/tP" : "speedup"), "effic."
    }
    cores = $2 * $3
    ratio = base / $11
    efficiency = (study == "weak") ? ratio : ratio / cores
    printf "  %-9s %-9s %-13s %8.1f ms %8.1f ms %9.2f %8.0f%%\n",
           $2 " x " $3, $4 "x" $5 "x" $6, $7 "x" $8 "x" $9, $11, $12,
           ratio, 100 * efficiency
}
' "$results"
