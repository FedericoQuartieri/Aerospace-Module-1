#!/usr/bin/env bash
#PBS -N nsb-threads
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Quanto scala il multithreading su una macchina vera.
#
#   qsub scripts/run_thread_scaling.sh        sul cluster
#   ./scripts/run_thread_scaling.sh           in locale, per provare
#
# ----------------------------------------------------------------------------
# Perche' misura DUE eseguibili
# ----------------------------------------------------------------------------
#
# Sono lo stesso solutore, con gli stessi kernel e la stessa griglia. Cambia
# una riga sola nello scenario:
#
#   bench   da test/paper_man.c, .porosity_time_dependent = 1
#           -> ricalcola la permeabilita' a OGNI passo temporale, tre
#              componenti su tutta la griglia, con i cicli SERIALI di
#              field.c, e senza che nessun cronometro la misuri
#
#   solver  da src/main.c con paper_data, .porosity_time_dependent = 0
#           -> quella riga la salta
#
# Le due colonne rispondono a domande diverse: la prima e' quanto scala il
# solutore mentre un quarto del lavoro resta seriale, la seconda e' quanto
# scalano i kernel threadati. La loro DIFFERENZA e' la misura di quanto costa
# quel ciclo, ed e' il risultato -- non un dettaglio metodologico.
#
# ----------------------------------------------------------------------------
# La colonna "non contato"
# ----------------------------------------------------------------------------
#
# Per ogni misura viene sommato il tempo dei singoli stadi e confrontato col
# totale. Cio' che avanza e' lavoro che nessuna statistica attribuisce a
# nessuno, e su paper_man e' fra un quarto e due terzi del passo.
#
# E' la colonna che va guardata per prima: finche' non torna, qualunque
# conclusione su quale stadio sia lento riguarda solo la parte cronometrata.
#
# ----------------------------------------------------------------------------
# Perche' ncpus=112 e non 56
# ----------------------------------------------------------------------------
#
# Il nodo ha 2 socket x 28 core x 2 thread SMT: 56 core fisici, 112 CPU
# logiche. PBS conta le logiche, e chiedendole tutte non ne resta nessuna per
# gli altri job: il nodo e' di fatto esclusivo. Serve, perche' la politica del
# cluster vieta place=excl e sulla coda `cpu' i nodi hanno abitualmente quattro
# o cinque job addosso -- e questo solutore, misurato accanto a loro, e' uscito
# 3.6 volte piu' lento senza che niente lo segnalasse.

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

read -r NX NY NZ <<< "${GRID:-256 256 256}"
STEPS="${STEPS:-10}"
REPEATS="${REPEATS:-2}"
THREADS="${THREADS:-1 2 4 7 14 28 56}"

build="build/threads"
results="$build/results.csv"
log="$build/run.log"
config="$build/grid.txt"
mkdir -p "$build"

# Tutto l'output finisce anche in un file dentro il repo: PBS consegna il
# proprio `.o<jobid>' dove decide la configurazione del sito, e su questo
# cluster non arriva ne' nella directory di sottomissione ne' nella home.
# Cosi' il log sta accanto al csv, e se il job viene ucciso dal walltime resta
# tutto quello che era stato stampato.
exec > >(tee "$log") 2>&1

# ------------------------------------------------------------------ macchina

echo "=== macchina ==="
physical=$(( $(lscpu | awk -F: '/Socket\(s\)/ {print $2}') * \
             $(lscpu | awk -F: '/Core\(s\) per socket/ {print $2}') ))
sockets=$(lscpu | awk -F: '/Socket\(s\)/ {gsub(/ /,"",$2); print $2}')
printf 'core fisici:   %s\n' "$physical"
printf 'cpu logiche:   %s\n' "$(nproc)"
printf 'nodi NUMA:     %s\n' \
    "$(find /sys/devices/system/node -maxdepth 1 -name 'node[0-9]*' | wc -l)"

# Il nodo e' davvero tutto nostro? Su questo cluster la risposta e' stata no
# due volte su tre, e le misure raccolte in quei casi erano inservibili senza
# che niente nell'output lo dicesse.
allowed="$(grep Cpus_allowed_list /proc/self/status | cut -f2)"
printf 'cpu concesse:  %s' "$allowed"
if [[ "$allowed" == "0-$(( $(nproc) - 1 ))" ]]; then
    printf '   (nodo intero)\n'
else
    printf '\n\n  ATTENZIONE: non hai tutto il nodo. Le misure che seguono\n'
    printf '  sono contaminate dai job dei vicini e non vanno usate.\n'
    printf '  Serve:  qsub -q scalability -l select=1:ncpus=%s\n\n' "$(nproc)"
fi
echo

# ----------------------------------------------------------------- compilazione

echo "=== compilazione (una volta sola) ==="
core_sources=()
for source in src/*.c; do
    [[ "$(basename -- "$source")" == "main.c" ]] || core_sources+=("$source")
done

cflags=(-std=gnu11 -O3 -Wall -Wextra -Iinclude
        -mavx2 -mfma -DUSE_SIMD -fopenmp -DUSE_OMP -DUSE_MPI)

# bench: la griglia e' una costante di compilazione perche' i test non leggono
# nessun file di configurazione.
printf '  bench  (paper_man, K ricalcolato ogni passo)   '
"${MPICC:-mpicc}" "${cflags[@]}" \
    -DDEFAULT_WIDTH="$NX" -DDEFAULT_HEIGHT="$NY" -DDEFAULT_DEPTH="$NZ" \
    -DDEFAULT_T=1e-1 -DDEFAULT_STEPS="$STEPS" \
    test/paper_man.c "${core_sources[@]}" -fopenmp -lm -o "$build/bench"
echo ok

# solver: la griglia la legge da file, quindi basta un binario.
printf '  solver (paper_data, K statico)                 '
"${MPICC:-mpicc}" "${cflags[@]}" \
    src/main.c "${core_sources[@]}" -fopenmp -lm -o "$build/solver"
echo ok

printf 'width = %s\nheight = %s\ndepth = %s\nsteps = %s\nt_end = 1e-1\n' \
    "$NX" "$NY" "$NZ" "$STEPS" > "$config"

echo "  griglia ${NX}x${NY}x${NZ}, $STEPS passi, $REPEATS ripetizioni"
echo

# -------------------------------------------------------------------- misura

printf 'caso,threads,wall_ms,eta,zeta,u,psi,philow,phihigh,pressure,somma_stadi,non_contato\n' \
    > "$results"

# Esegue una configurazione, tiene la ripetizione migliore, e scompone il
# tempo. La somma degli stadi confrontata col totale e' l'unica cosa che
# rivela il lavoro che nessuno cronometra.
measure()
{
    local caso="$1" threads="$2"
    shift 2

    local best_wall="" best_line=""
    for ((r = 0; r < REPEATS; r++)); do
        # --bind-to none e' obbligatorio, non un dettaglio: per default mpirun
        # inchioda il processo a UN core, e i suoi thread se lo spartiscono
        # invece di prendersene uno a testa. Senza, la colonna dei thread
        # misura zero guadagno e niente nell'output lo segnala.
        local out
        out="$(OMP_NUM_THREADS="$threads" OMP_PLACES="${PLACES:-cores}" \
               OMP_PROC_BIND="${BIND:-close}" \
               "$@" mpirun --bind-to none -n 1 "$exe" $exe_args 2>&1)"

        # Pattern ancorati: "eta system" senza ancora matcherebbe anche
        # "zeta system", e si finirebbe a leggere i tempi di zeta credendoli
        # di eta -- numeri plausibili, di un'altra cosa.
        local line
        line="$(awk '
            /^  eta system/    {eta   = $3}
            /^  zeta system/   {zeta  = $3}
            /^  u system/      {u     = $3}
            /^  psi system/    {psi   = $3}
            /^  phi low/       {lo    = $3}
            /^  phi high/      {hi    = $3}
            /^  pressure:/     {pr    = $3}
            /wall per step:/   {wall  = $4}
            END {
                sum = eta + zeta + u + psi + lo + hi + pr
                printf "%s,%s,%s,%s,%s,%s,%s,%s,%.3f,%.3f",
                       wall, eta, zeta, u, psi, lo, hi, pr, sum, wall - sum
            }' <<< "$out")"

        local wall="${line%%,*}"
        if [[ -z "$best_wall" ]] || awk "BEGIN{exit !($wall < $best_wall)}"; then
            best_wall="$wall"
            best_line="$line"
        fi
    done

    printf '%s,%s,%s\n' "$caso" "$threads" "$best_line" >> "$results"

    local untimed="${best_line##*,}"
    printf '%9s ms   (non contato: %s ms)\n' "$best_wall" "$untimed"
}

echo "=== paper_man: K ricalcolato a ogni passo ==="
exe="$build/bench"; exe_args=""
for t in $THREADS; do
    printf '  %3s thread  ' "$t"
    measure paper_man "$t"
done

echo
echo "=== paper_data: K statico ==="
exe="$build/solver"; exe_args="$config"
for t in $THREADS; do
    printf '  %3s thread  ' "$t"
    measure paper_data "$t"
done

# --- piazzamento sui socket -------------------------------------------------
#
# Stessi thread, stessa potenza di calcolo, banda diversa: meta' dei core su un
# socket solo vuol dire 6 canali di memoria, distribuiti sui due ne fa 12.
#
# Va misurato su paper_data e non su paper_man: sul secondo il ciclo seriale
# domina il passo e coprirebbe qualunque effetto della banda -- e' gia'
# successo, e la prima volta la risposta era stata "nessuna differenza" per
# quel motivo e non perche' la banda non contasse.
half=$(( physical / sockets ))
if command -v numactl > /dev/null && [[ "$sockets" -gt 1 ]]; then
    echo
    echo "=== piazzamento: $half thread, un socket contro due ==="
    exe="$build/solver"; exe_args="$config"

    printf '  un socket  (%2s canali di memoria)   ' "6"
    measure socket_singolo "$half" numactl --cpunodebind=0 --membind=0

    printf '  due socket (%2s canali di memoria)   ' "12"
    BIND=spread measure socket_doppio "$half"
fi

# ------------------------------------------------------------------ risultato

echo
echo "=== risultato ==="
awk -F, -v physical="$physical" '
NR == 1 { next }
{
    caso[$1]; wall[$1 "," $2] = $3; untimed[$1 "," $2] = $12
    if (!($2 in seen)) { order[++n] = $2; seen[$2] = 1 }
}
END {
    printf "\n  %-7s %12s %8s %8s %10s   %12s %8s %8s %10s\n",
           "", "paper_man", "", "", "", "paper_data", "", "", ""
    printf "  %-7s %12s %8s %8s %10s   %12s %8s %8s %10s\n",
           "thread", "ms/passo", "speedup", "effic.", "non cont.",
           "ms/passo", "speedup", "effic.", "non cont."
    for (i = 1; i <= n; i++) {
        t = order[i]
        a = wall["paper_man," t]; b = wall["paper_data," t]
        if (i == 1) { base_a = a; base_b = b }
        sa = (a > 0) ? base_a / a : 0
        sb = (b > 0) ? base_b / b : 0
        printf "  %-7s %12.1f %7.2fx %7.0f%% %9.0f   %12.1f %7.2fx %7.0f%% %9.0f\n",
               t, a, sa, 100 * sa / t, untimed["paper_man," t],
                  b, sb, 100 * sb / t, untimed["paper_data," t]
    }
    if ("socket_singolo,"  physical / 2 in wall) { }
    for (k in wall) {
        split(k, p, ",")
        if (p[1] ~ /^socket_/) {
            printf "\n  %-22s %10.1f ms", p[1], wall[k]
        }
    }
    printf "\n"
}
' "$results"

cat <<'NOTA'

  La colonna "non cont." e' il tempo del passo che nessuno stadio dichiara:
  totale meno la somma dei cronometri. Su paper_man e' il ciclo seriale che
  ricalcola la permeabilita' a ogni passo (solver.c:105 -> field.c), e siccome
  non si accorcia coi thread e' lui a fissare il tetto dello speedup.

NOTA

echo "csv in $results"
echo "log in $log"
