#!/usr/bin/env bash
#PBS -N nsb-diagnosi
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Perche' le configurazioni ibride crollano: quale delle tre cause e'.
#
#   qsub scripts/run_hybrid_diagnosi.sh       sul cluster
#   ./scripts/run_hybrid_diagnosi.sh          in locale, per provare
#
# ----------------------------------------------------------------------------
# La domanda
# ----------------------------------------------------------------------------
#
# run_hybrid.sh ha misurato che a parita' di core (rank x thread = 56) le due
# estremita' vanno uguali e tutto il centro e' fino a 14 volte piu' lento:
#
#     1 x 56   274 ms        8 x 7   2021 ms
#     2 x 28  3865 ms       14 x 4   1760 ms
#     4 x 14  3746 ms       28 x 2    814 ms
#                           56 x 1    287 ms
#
# La spiegazione scritta in MULTITHREAD.md e' il costo di aprire un team di
# thread per ogni piano. Non regge l'aritmetica: a 256^3 con np=4 le aperture
# per passo sono circa 4100, e 3459 ms / 4100 fa 844 us per apertura. Un
# fork/join misurato costa circa 1 us, e anche a 28 thread su due socket non
# arriva a 20. Mancano due ordini di grandezza.
#
# Nemmeno le collettive spiegano la forma: 4x14 ne fa 2048 per passo con 4
# partecipanti, 56x1 ne fa 1280 con 56 partecipanti, e 56x1 e' 13 volte piu'
# VELOCE. Se il costo stesse li', l'ordine sarebbe rovesciato.
#
# Restano tre ipotesi, e questo script le separa. Nessuna richiede di
# modificare il solutore.
#
#   A  spin di OpenMP.  Finita una regione parallela i worker di libgomp
#      restano a girare a vuoto prima di addormentarsi. Il master intanto
#      entra in una collettiva bloccante. Su un nodo saturo quegli spinner
#      tolgono la CPU ai master degli altri rank, e la collettiva -- che e'
#      sincronizzante, quindi aspetta il piu' lento -- paga un quanto di
#      scheduling. 3459 ms / 2048 collettive = 1.7 ms, che e' l'ordine di
#      grandezza di un timeslice del CFS: e' il motivo per cui questa ipotesi
#      e' la prima della lista.
#      Si toglie con OMP_WAIT_POLICY=passive.
#
#   B  busy-wait di MPI.  Open MPI dentro una chiamata bloccante fa polling
#      aggressivo invece di cedere la CPU, con lo stesso effetto di A ma dal
#      lato opposto.
#      Si toglie con --mca mpi_yield_when_idle 1.
#
#   C  costo intrinseco dei thread, indipendente da chi altro gira sul nodo.
#      Se fosse questo, ne' A ne' B cambierebbero niente e il tempo crescerebbe
#      coi thread anche a nodo mezzo vuoto.
#
# ----------------------------------------------------------------------------
# Perche' rank fisso a 4 e thread variabili
# ----------------------------------------------------------------------------
#
# La tabella di run_hybrid.sh tiene rank x thread = 56, quindi ogni riga cambia
# TRE cose insieme: quanti thread per rank, quanto e' saturo il nodo, e la
# forma della decomposizione (MPI_Dims_create da 2x1x1 a 7x4x2). Da quella
# tabella le tre non si separano, ed e' il motivo per cui la causa e' rimasta
# incerta.
#
# Qui il numero di rank resta 4 in tutta la serie centrale. La decomposizione
# e' quindi sempre 2x2x1, i piani sono sempre gli stessi, le collettive sono
# sempre 2048 per passo con 4 partecipanti. L'unica cosa che cambia sono i
# thread per rank -- e con essi la saturazione del nodo, che a 4x7 e' al 50%
# e a 4x14 al 100%.
#
# Se il tempo cresce lungo quella serie, cresce per una ragione che non ha
# niente a che vedere con MPI, perche' il lavoro MPI e' identico in tutte.
#
# ----------------------------------------------------------------------------
# Perche' questo risponde anche sul multi-nodo
# ----------------------------------------------------------------------------
#
# 4 rank x 14 thread su UN nodo da 56 core e 4 rank x 14 thread su QUATTRO nodi
# da 56 core sono la stessa decomposizione, le stesse collettive, lo stesso
# lavoro per rank. Cambiano due cose sole: le collettive passano dalla memoria
# condivisa alla rete (peggio), e ogni rank smette di condividere i core con
# gli altri tre (meglio).
#
# 4x7 su un nodo intero e' il surrogato di quel secondo effetto: 28 thread su
# 56 core, con 28 core liberi. Se il crollo sparisce li', e' un problema di
# nodo saturo -- e il multi-nodo non lo erediterebbe, lo cancellerebbe.
# Se il crollo resta, e' del codice, e allora il multi-nodo se lo porta dietro.
#
# ----------------------------------------------------------------------------
# La colonna che decide
# ----------------------------------------------------------------------------
#
# utils.c stampa gia' `mpi per step': i nanosecondi passati DENTRO le chiamate
# MPI, cronometrate una per una in parallel.c. Non serve strumentare niente.
#
#   mpi_ms vicino a wall_ms   il tempo e' dentro le collettive (A o B)
#   mpi_ms trascurabile       il tempo e' fuori, nelle regioni OpenMP (C)
#
# E' la prima colonna da leggere: da sola divide le tre ipotesi in due gruppi,
# e i casi passive/yield dicono quale dei due.

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

read -r NX NY NZ <<< "${GRID:-256 256 256}"
STEPS="${STEPS:-10}"
REPEATS="${REPEATS:-1}"

build="build/diagnosi"
results="$build/results.csv"
log="$build/run.log"
config="$build/grid.txt"
mkdir -p "$build"

exec > >(tee "$log") 2>&1

# ------------------------------------------------------------------- branch

# Il multithreading non e' su tutti i branch: `mpi', `fast' e `pipeline' non
# hanno ne' workers.h ne' la condizione whole_axis, e su `unified' il backend
# Schur e' stato spostato in src/tridiag/schur/ mentre questo script -- come
# run_hybrid.sh -- compila il glob piatto src/*.c. Scoprirlo qui costa un
# secondo, scoprirlo dall'errore del linker dentro il job costa una coda.
echo "=== branch ==="
printf 'HEAD:          %s\n' "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '(non un repo git)')"

if ! grep -q whole_axis src/momentum.c 2> /dev/null; then
    cat << 'SBAGLIATO'

ERRORE: src/momentum.c non contiene la condizione `whole_axis', quindi questo
albero non ha il multithreading e non c'e' niente da diagnosticare.

Il lavoro sui thread sta sul branch `multithread', che e' anche quello da cui
viene la tabella rank x thread di MULTITHREAD.md 9.2. I branch `mpi', `fast' e
`pipeline' non ce l'hanno affatto.

  git checkout multithread

Su `unified' il codice c'e' ma sta in src/tridiag/schur/: il glob src/*.c di
questo script (e di run_hybrid.sh) non lo raccoglie e il link fallisce. Le
misure vanno fatte comunque su `multithread', perche' e' li' che sono stati
misurati i 3746 ms con cui vanno confrontate.

SBAGLIATO
    exit 1
fi
echo

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
    printf '\n\n  ATTENZIONE: non hai tutto il nodo. Tutta la diagnosi qui sotto\n'
    printf '  riguarda la contesa per i core: misurata accanto ai job di altri\n'
    printf '  utenti non significa niente.\n'
    printf '  Serve:  qsub -q scalability -l select=1:ncpus=%s\n\n' "$(nproc)"
fi

# In locale (portatile a 4 core) la matrice a 56 thread non ha senso: si
# riduce tutto in proporzione, cosi' lo script si prova prima di sottometterlo.
# RANK e' il perno di tutta la serie centrale: tenendolo fisso, decomposizione,
# piani e collettive restano identici e l'unica variabile sono i thread.
full="$physical"
if [[ "$physical" -ge 32 ]]; then
    RANK="${RANK:-4}"
else
    # In locale 4 rank x 14 thread non ci stanno: si scala il perno, cosi' lo
    # script si prova per intero prima di occupare una coda.
    RANK="${RANK:-2}"
    echo
    echo "  Nodo piccolo: perno a $RANK rank e griglia ridotta. I numeri"
    echo "  servono a vedere che lo script gira, non a rispondere."
    [[ "${GRID:-}" ]] || read -r NX NY NZ <<< "64 64 64"
fi
pieno=$(( physical / RANK ))                 # 14 sul nodo: RANK x pieno = saturo
meta=$(( pieno / 2 > 0 ? pieno / 2 : 1 ))    #  7 sul nodo: meta' nodo libero
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

printf 'caso,rank,thread,wait,yield,wall_ms,mpi_ms,mpi_pct,non_contato\n' > "$results"

# Stessa logica di piazzamento di run_hybrid.sh, e per la stessa ragione: con
# --bind-to none i thread di rank diversi si mescolerebbero sui due socket e la
# misura direbbe quanto costa quel disastro invece di quello che si cerca.
#
# L'eccezione resta il rank singolo, che deve coprire entrambi i socket e a cui
# il piazzamento lo da' OpenMP con PROC_BIND=spread.
measure()
{
    local caso="$1" ranks="$2" threads="$3" wait_policy="$4" yield="$5"
    local mapping place mca=()

    # mpirun rifiuta di legare piu' PE dei core disponibili, e il messaggio che
    # stampa non dice quale caso stava provando.
    if [[ $(( ranks * threads )) -gt "$physical" ]]; then
        printf 'saltato: %s x %s = %s PE, ma i core sono %s\n' \
            "$ranks" "$threads" "$(( ranks * threads ))" "$physical"
        return 0
    fi

    if [[ "$ranks" -eq 1 ]]; then
        mapping="--bind-to none"
        place="spread"
    else
        mapping="--map-by ppr:$(( ranks / sockets )):socket:PE=$threads --bind-to core"
        place="close"
    fi

    [[ "$yield" == "si" ]] && mca=(--mca mpi_yield_when_idle 1)

    local best_wall="" best_line=""
    for ((r = 0; r < REPEATS; r++)); do
        local out
        out="$(OMP_NUM_THREADS="$threads" OMP_PLACES=cores OMP_PROC_BIND="$place" \
               OMP_WAIT_POLICY="$wait_policy" \
               mpirun $mapping "${mca[@]}" -n "$ranks" \
               "$build/solver" "$config" 2>&1)" || true

        local line
        line="$(awk '
            /^  eta system/  {eta = $3}
            /^  zeta system/ {zeta = $3}
            /^  u system/    {u   = $3}
            /^  psi system/  {psi = $3}
            /^  phi low/     {lo  = $3}
            /^  phi high/    {hi  = $3}
            /^  pressure:/   {pr  = $3}
            /^  wall per step:/ {wall = $4}
            /^  mpi per step:/  {mpi = $4}
            END {
                sum = eta + zeta + u + psi + lo + hi + pr
                pct = (wall > 0) ? 100 * mpi / wall : 0
                printf "%s,%s,%.1f,%.3f", wall, mpi, pct, wall - sum
            }' <<< "$out")"

        local wall="${line%%,*}"
        if [[ -z "$wall" ]]; then
            echo "    nessun tempo nell'output -- ecco cosa ha stampato:"
            sed 's/^/      /' <<< "$out" | head -20
            return 1
        fi
        if [[ -z "$best_wall" ]] || awk "BEGIN{exit !($wall < $best_wall)}"; then
            best_wall="$wall"; best_line="$line"
        fi
    done

    printf '%s,%s,%s,%s,%s,%s\n' \
        "$caso" "$ranks" "$threads" "$wait_policy" "$yield" "$best_line" >> "$results"

    local mpi_ms; mpi_ms="$(cut -d, -f2 <<< "$best_line")"
    local pct;    pct="$(cut -d, -f3 <<< "$best_line")"
    printf '%9s ms   (dentro MPI: %s ms = %s%%)\n' "$best_wall" "$mpi_ms" "$pct"
}

# --- 1. le due estremita', per ancorare tutto il resto ----------------------
#
# Se questi due non riproducono i 274 e i 287 ms di run_hybrid.sh, qualcosa e'
# cambiato fra allora e adesso e il resto della tabella non va confrontato con
# quella vecchia.

echo "=== ancore: le due estremita' gia' misurate ==="
printf '  1 x %-3s (0 collettive)        ' "$full"
measure ancora_thread 1 "$full" active no || true
printf '  %-3s x 1 (0 team di thread)    ' "$full"
measure ancora_rank "$full" 1 active no || true

# --- 2. la serie a rank fisso ----------------------------------------------
#
# Decomposizione, piani e collettive identici in tutte e quattro: cambia solo
# il numero di thread per rank, e con esso la saturazione del nodo.

echo
echo "=== rank fisso a $RANK: stesse collettive, thread variabili ==="
# Su un nodo piccolo meta' e pieno possono coincidere con 1 o 2: senza togliere
# i doppioni la stessa configurazione finirebbe due volte nel csv.
serie=""
for t in 1 2 "$meta" "$pieno"; do
    [[ " $serie " == *" $t "* ]] || serie="$serie $t"
done

for t in $serie; do
    printf '  %s x %-3s (nodo al %3s%%)        ' \
        "$RANK" "$t" "$(( 100 * RANK * t / physical ))"
    measure "fisso_t$t" "$RANK" "$t" active no || true
done

echo
echo "=== stesso caso $RANK x $pieno, togliendo una causa alla volta ==="
printf '  OMP_WAIT_POLICY=passive       '
measure cura_omp_passive "$RANK" "$pieno" passive no || true
printf '  mpi_yield_when_idle=1         '
measure cura_mpi_yield "$RANK" "$pieno" active si || true
printf '  tutte e due                   '
measure cura_entrambe "$RANK" "$pieno" passive si || true

# Il 2x28 e' l'altra estremita' del crollo (3865 ms): se la cura funziona la'
# dove il team e' il doppio, la lettura regge; se funziona solo su 4x14 no.
if [[ "$sockets" -ge 2 && "$physical" -ge 32 ]]; then
    echo
    echo "=== controprova su 2 x $(( physical / 2 )), l'altro caso rotto ==="
    printf '  base                          '
    measure due_rank_base 2 $(( physical / 2 )) active no || true
    printf '  con entrambe le cure          '
    measure due_rank_cura 2 $(( physical / 2 )) passive si || true
fi

# ------------------------------------------------------------------ verdetto

echo
echo "=== tabella ==="
echo
printf '  %-20s %7s %-9s %-7s %11s %11s %7s\n' \
    caso "rxt" wait mpi "wall (ms)" "in MPI (ms)" "in MPI"
awk -F, 'NR > 1 {
    printf "  %-20s %3sx%-3s %-9s %-7s %11.1f %11.1f %6s%%\n",
           $1, $2, $3, $4, ($5 == "si" ? "yield" : "-"), $6, $7, $8
}' "$results"

# Le tre ipotesi si distinguono da quattro rapporti, e ognuno isola una cosa
# sola. Il testo qui sotto li calcola invece di lasciarli da fare a mano,
# perche' e' esattamente il passaggio in cui la lettura precedente si era
# sbagliata: guardare la tabella e concludere senza dividere.
awk -F, -v pieno="$pieno" -v meta="$meta" -v rank="$RANK" '
NR > 1 { wall[$1] = $6 + 0; mpi[$1] = $7 + 0 }
END {
    base = wall["fisso_t" pieno]
    mez  = wall["fisso_t" meta]
    uno  = wall["fisso_t1"]

    print ""
    print "=== lettura ==="
    print ""

    if (base <= 0) {
        print "  Il caso di riferimento non ha prodotto un tempo: niente da leggere."
        exit
    }

    frazione = 100 * mpi["fisso_t" pieno] / base
    printf "  1. Dove sta il tempo di %dx%d: %.0f%% dentro le chiamate MPI.\n", rank, pieno, frazione
    if (frazione > 60)
        print "     -> si accumula nelle collettive, non nelle regioni OpenMP."
    else if (frazione < 20)
        print "     -> le collettive sono innocenti: il tempo e fuori da MPI (ipotesi C)."
    else
        print "     -> nessuno dei due domina: il costo e ripartito."

    if (mez > 0) {
        printf "\n  2. Saturazione: %dx%d (nodo pieno) = %.0f ms contro %dx%d (meta nodo) = %.0f ms, %.1fx.\n",
               rank, pieno, base, rank, meta, mez, base / mez
        if (base / mez > 3)
            print "     -> il crollo vuole il nodo saturo. Su nodi separati ogni rank\n        avrebbe i core per se: il MULTI-NODO NON LO EREDITEREBBE."
        else
            print "     -> il crollo c e anche a nodo mezzo vuoto: e del codice, e il\n        MULTI-NODO SE LO PORTEREBBE DIETRO."
    }

    print ""
    if (wall["cura_omp_passive"] > 0) {
        printf "  3. OMP_WAIT_POLICY=passive: %.0f -> %.0f ms (%.1fx)", base, wall["cura_omp_passive"], base / wall["cura_omp_passive"]
        print (base / wall["cura_omp_passive"] > 2) ? "   IPOTESI A: erano i thread OpenMP a vuoto." : ""
    }
    if (wall["cura_mpi_yield"] > 0) {
        printf "     mpi_yield_when_idle=1:   %.0f -> %.0f ms (%.1fx)", base, wall["cura_mpi_yield"], base / wall["cura_mpi_yield"]
        print (base / wall["cura_mpi_yield"] > 2) ? "   IPOTESI B: era il polling di Open MPI." : ""
    }
    if (wall["cura_entrambe"] > 0)
        printf "     entrambe:                %.0f -> %.0f ms (%.1fx)\n", base, wall["cura_entrambe"], base / wall["cura_entrambe"]

    if (uno > 0) {
        printf "\n  4. A %d rank fissi le collettive per passo sono le stesse in ogni riga:\n", rank
        printf "     1 thread = %.0f ms, %d thread = %.0f ms, cioe %.1fx. Quel fattore non\n", uno, pieno, base, base / uno
        print  "     puo venire da MPI, perche il lavoro MPI e identico nelle due."
    }

    if (wall["due_rank_base"] > 0 && wall["due_rank_cura"] > 0)
        printf "\n  5. Controprova su 2 rank (team doppio): %.0f -> %.0f ms (%.1fx).\n",
               wall["due_rank_base"], wall["due_rank_cura"], wall["due_rank_base"] / wall["due_rank_cura"]
    print ""
}' "$results"

cat << 'NOTA'
  Se nessuna cura sposta niente E il crollo resta a nodo mezzo vuoto, resta in
  piedi solo l ipotesi C: il costo e nel codice. Il passo dopo e' allora
  profilare una singola momentum_direction con perf, non aggiungere altre
  variabili d ambiente.

NOTA

echo "csv in $results"
echo "log in $log"
