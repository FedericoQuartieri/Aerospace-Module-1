#!/usr/bin/env bash
#PBS -N nsb-hybrid
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# A parita' di core occupati, conviene spenderli in processi o in thread?
#
#   qsub scripts/run_hybrid.sh        sul cluster
#   ./scripts/run_hybrid.sh           in locale, per provare
#
# ----------------------------------------------------------------------------
# Perche' la domanda non e' oziosa
# ----------------------------------------------------------------------------
#
# Un processo prende un blocco di dominio, e appena una direzione viene divisa
# succedono due cose: i kernel vettorizzati di quella direzione smettono di
# applicarsi (valgono solo su una linea intera) e il complemento di Schur passa
# da una risoluzione locale per linea a tre.
#
# Un thread prende delle linee, che sono indipendenti comunque sia diviso il
# dominio, e non paga ne' l'una ne' l'altra cosa.
#
# Su un portatile a un socket la risposta era netta: 1 processo x 4 thread
# batteva 4 processi x 1 thread del 17%. Qui i socket sono due, e la risposta
# potrebbe ribaltarsi: 1x56 deve attraversare entrambi i socket e paga la
# latenza NUMA, mentre 2x28 mette un rank per socket con localita' perfetta.
# E' esattamente per questo che va misurato invece che dedotto.
#
# ----------------------------------------------------------------------------
# Perche' paper_data e non paper_man
# ----------------------------------------------------------------------------
#
# paper_man ricalcola la permeabilita' a ogni passo con i cicli seriali di
# field.c: un lavoro che non si accorcia coi thread e che, secondo le misure
# di run_thread_scaling.sh, vale fra un quarto e due terzi del passo. Coprirebbe
# proprio l'effetto che questo studio cerca. paper_data ha K statico e salta
# quella riga.
#
# ----------------------------------------------------------------------------
# Il piazzamento, che qui e' meta' del risultato
# ----------------------------------------------------------------------------
#
# Con piu' di un rank ogni rank prende un socket intero e i suoi thread stanno
# li' dentro (--map-by ppr:N:socket:PE=T --bind-to core). Con --bind-to none i
# thread di rank diversi si mescolerebbero sui due socket, e la misura direbbe
# quanto costa quel disastro invece che quanto costa dividere il dominio.
#
# Il caso a un rank e' l'eccezione: deve coprire entrambi i socket, quindi si
# lascia bindare a OpenMP con PROC_BIND=spread, che e' cio' che distribuisce i
# thread su tutti e 12 i canali di memoria invece che sui 6 di un socket solo.
#
# ----------------------------------------------------------------------------
# Il multi-nodo non c'e', e non e' una svista
# ----------------------------------------------------------------------------
#
# Su questo cluster non esiste un modo funzionante di lanciare processi su piu'
# nodi: l'Open MPI di sistema non ha il componente `plm tm' (solo rsh, slurm,
# isolated), sshd sui compute non legge ~/.ssh/authorized_keys, e pbs_tmrsh
# risponde `tm_poll(obit): err not connected'. Piu' rank sullo STESSO nodo
# invece funzionano, ed e' quello che misura questo script.

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

read -r NX NY NZ <<< "${GRID:-256 256 256}"
STEPS="${STEPS:-10}"
REPEATS="${REPEATS:-2}"

# rank x thread, con il prodotto sempre uguale ai core fisici. 7 rank sono
# esclusi apposta: non si dividono su due socket senza spezzarne uno.
CONFIGS="${CONFIGS:-1x56 2x28 4x14 8x7 14x4 28x2 56x1}"

build="build/hybrid"
results="$build/results.csv"
log="$build/run.log"
config="$build/grid.txt"
mkdir -p "$build"

# Come in run_thread_scaling.sh: su questo sito il .o<jobid> di PBS non arriva
# ne' nella directory di sottomissione ne' nella home.
exec > >(tee "$log") 2>&1

# ------------------------------------------------------------------ macchina

echo "=== macchina ==="
sockets=$(lscpu | awk -F: '/Socket\(s\)/ {gsub(/ /,"",$2); print $2}')
per_socket=$(lscpu | awk -F: '/Core\(s\) per socket/ {gsub(/ /,"",$2); print $2}')
physical=$(( sockets * per_socket ))
printf 'socket:        %s\n' "$sockets"
printf 'core fisici:   %s\n' "$physical"
printf 'cpu logiche:   %s\n' "$(nproc)"

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

echo "=== compilazione ==="
core_sources=()
for source in src/*.c; do
    [[ "$(basename -- "$source")" == "main.c" ]] || core_sources+=("$source")
done

printf '  solver (paper_data, K statico)   '
"${MPICC:-mpicc}" -std=gnu11 -O3 -Wall -Wextra -Iinclude \
    -mavx2 -mfma -DUSE_SIMD -fopenmp -DUSE_OMP -DUSE_MPI \
    src/main.c "${core_sources[@]}" -fopenmp -lm -o "$build/solver"
echo ok

printf 'width = %s\nheight = %s\ndepth = %s\nsteps = %s\nt_end = 1e-1\n' \
    "$NX" "$NY" "$NZ" "$STEPS" > "$config"
echo "  griglia ${NX}x${NY}x${NZ}, $STEPS passi, $REPEATS ripetizioni"
echo

# -------------------------------------------------------------------- misura

printf 'rank,thread,wall_ms,eta,zeta,u,psi,philow,phihigh,pressure,somma_stadi,non_contato\n' \
    > "$results"

measure()
{
    local ranks="$1" threads="$2"
    local bind place mapping

    if [[ "$ranks" -eq 1 ]]; then
        # Un rank solo: deve coprire entrambi i socket, e a distribuirlo e'
        # OpenMP. --bind-to none serve a togliere di mezzo MPI, che altrimenti
        # inchioderebbe il processo a un core e i thread se lo spartirebbero.
        mapping="--bind-to none"
        place="spread"
    else
        mapping="--map-by ppr:$(( ranks / sockets )):socket:PE=$threads --bind-to core"
        place="close"
    fi
    bind="$place"

    local best_wall="" best_line=""
    for ((r = 0; r < REPEATS; r++)); do
        local out
        out="$(OMP_NUM_THREADS="$threads" OMP_PLACES=cores OMP_PROC_BIND="$bind" \
               mpirun $mapping -n "$ranks" "$build/solver" "$config" 2>&1)"

        # Pattern ancorati: "eta system" senza ancora matcherebbe anche
        # "zeta system", e si leggerebbero i tempi di zeta credendoli di eta.
        local line
        line="$(awk '
            /^  eta system/    {eta  = $3}
            /^  zeta system/   {zeta = $3}
            /^  u system/      {u    = $3}
            /^  psi system/    {psi  = $3}
            /^  phi low/       {lo   = $3}
            /^  phi high/      {hi   = $3}
            /^  pressure:/     {pr   = $3}
            /wall per step:/   {wall = $4}
            END {
                sum = eta + zeta + u + psi + lo + hi + pr
                printf "%s,%s,%s,%s,%s,%s,%s,%s,%.3f,%.3f",
                       wall, eta, zeta, u, psi, lo, hi, pr, sum, wall - sum
            }' <<< "$out")"

        local wall="${line%%,*}"
        if [[ -z "$wall" ]]; then
            echo "    nessun tempo nell'output -- ecco cosa ha stampato:"
            sed 's/^/      /' <<< "$out" | head -20
            return 1
        fi
        if [[ -z "$best_wall" ]] || awk "BEGIN{exit !($wall < $best_wall)}"; then
            best_wall="$wall"
            best_line="$line"
        fi
    done

    printf '%s,%s,%s\n' "$ranks" "$threads" "$best_line" >> "$results"
    printf '%9s ms   (non contato: %s ms)\n' "$best_wall" "${best_line##*,}"
}

echo "=== rank x thread, prodotto costante = $physical core ==="
for spec in $CONFIGS; do
    ranks="${spec%x*}"
    threads="${spec#*x}"
    if [[ "$ranks" -gt 1 && $(( ranks % sockets )) -ne 0 ]]; then
        printf '  %-7s  saltata: %s rank non si dividono su %s socket\n' \
            "$spec" "$ranks" "$sockets"
        continue
    fi
    # Il senso dello studio e' confrontare a parita' di core occupati: una
    # riga che ne usa un numero diverso non e' confrontabile con le altre, e
    # se ne usa di piu' misura una macchina in sovraccarico.
    if [[ $(( ranks * threads )) -ne "$physical" ]]; then
        printf '  %-7s  saltata: %s core invece dei %s fisici\n' \
            "$spec" "$(( ranks * threads ))" "$physical"
        continue
    fi
    printf '  %-7s  ' "$spec"
    measure "$ranks" "$threads"
done

# ------------------------------------------------------------------ risultato

echo
echo "=== risultato ==="
awk -F, '
NR == 1 { next }
{
    n++
    rk[n] = $1; th[n] = $2; wall[n] = $3; unt[n] = $12
    if (n == 1) base = $3
}
END {
    printf "\n  %-9s %12s %9s %12s\n",
           "rank x th", "ms/passo", "vs " rk[1] "x" th[1], "non contato"
    for (i = 1; i <= n; i++) {
        printf "  %-9s %12.1f %8.2fx %12.0f\n",
               rk[i] " x " th[i], wall[i], base / wall[i], unt[i]
    }
}
' "$results"

cat << 'NOTA'

  La colonna "vs ..." e' il rapporto col primo caso, non uno speedup: tutte
  le righe usano lo stesso numero di core, quindi qui non c'e' niente da
  dividere per il parallelismo. Sopra 1.00 vuol dire che spezzare il dominio
  in quel modo conviene, sotto vuol dire che costa.

  "non contato" e' il tempo del passo che nessuno stadio dichiara: totale
  meno la somma dei cronometri. Su paper_data dovrebbe essere piccolo; se non
  lo e', prima di leggere le altre colonne conviene capire perche'.

NOTA

echo "csv in $results"
echo "log in $log"
