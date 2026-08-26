#!/usr/bin/env bash
#PBS -N nsb-multinode-check
#PBS -q cpu
#PBS -l select=4:ncpus=7
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -j oe
#
# Non e' uno studio di scaling. E' una verifica di correttezza: che lo
# scambio halo dia lo stesso risultato quando attraversa la rete fra nodi
# fisici invece della RAM condivisa di un socket.
#
#   qsub scripts/run_multinode_check.sh       sul cluster
#   ./scripts/run_multinode_check.sh          in locale, per provare (np=1)
#
# ----------------------------------------------------------------------------
# Perche' non e' un test di velocita'
# ----------------------------------------------------------------------------
#
# La coda `cpu' concede al massimo 28 CPU per job, sommate su tutti i chunk
# della select (verificato: select=2:ncpus=28 viene rifiutato, select=4:ncpus=7
# no). Con 4 nodi restano 7 CPU logiche a testa, su nodi che la coda non
# garantisce esclusivi. Qualunque tempo misurato qui e' contaminato dai
# vicini e non va confrontato con quelli di run_thread_scaling.sh.
#
# Quello che invece non dipende da quanti vicini hai e' se il risultato resta
# lo stesso. E' la stessa domanda che gia' verifica `paper_man` fra 1 e 8
# processi -- qui la novita' e' che i processi non condividono piu' la RAM.
#
# ----------------------------------------------------------------------------
# Perche' c'e' place=scatter
# ----------------------------------------------------------------------------
#
# Senza dirglielo, lo scheduler puo' impacchettare tutti e 4 i chunk sullo
# stesso nodo se quello ha abbastanza CPU libere -- ed e' esattamente quello
# che ha fatto la prima volta: 4x7 CPU, tutte su cpu02 da solo, "nodi
# distinti: 1". place=scatter obbliga un chunk per host, che e' il punto
# dell'intero script: senza, non si sta testando niente di diverso da
# run_thread_scaling.sh.
#
# ----------------------------------------------------------------------------
# Cosa viene confrontato
# ----------------------------------------------------------------------------
#
#   riferimento   1 rank, nessuna comunicazione: la verita' di base
#   multi-nodo    4 rank, uno per nodo, forma 1x2x2: tutti e tre gli assi
#                 divisi, quindi tutte e tre le direzioni di halo attraversano
#                 la rete almeno una volta
#
# Il confronto e' sulle 10 cifre che paper_man stampa di suo, la stessa
# precisione su cui si basa gia' l'affermazione "identico da 1 a 8 processi"
# nel README. Oltre quella precisione le somme ridotte fra processi ballano
# nelle ultime cifre anche in locale, su RAM condivisa: non e' un difetto
# della rete, quindi non e' quello che questo script deve stanare.
#
# Il multi-nodo viene eseguito due volte: se le due ripetizioni non
# coincidessero fra loro, prima ancora che col riferimento, il sospetto
# sarebbe un ordine di arrivo dei messaggi che il codice assume fisso e la
# rete non garantisce -- cosa che su RAM condivisa non si vedrebbe mai.

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

read -r NX NY NZ <<< "${GRID:-96 96 96}"
STEPS="${STEPS:-10}"

build="build/multinode"
log="$build/run.log"
mkdir -p "$build"

# Stesso motivo del run_thread_scaling.sh: su questo sito il file .o<jobid>
# di PBS non arriva ne' nella directory di sottomissione ne' nella home.
exec > >(tee "$log") 2>&1

# ------------------------------------------------------------------ nodi

echo "=== nodi assegnati dal job ==="
if [[ -n "${PBS_NODEFILE:-}" && -f "$PBS_NODEFILE" ]]; then
    sort -u "$PBS_NODEFILE"
    n_unique="$(sort -u "$PBS_NODEFILE" | wc -l)"
    echo "nodi distinti: $n_unique"
    if [[ "$n_unique" -lt 2 ]]; then
        echo
        echo "ATTENZIONE: un solo nodo assegnato. Questo script esiste per"
        echo "far attraversare l'halo dalla rete: su un nodo solo non prova"
        echo "niente che run_thread_scaling.sh non provi gia'. Mi fermo."
        exit 1
    fi
    echo

    # mpirun lancia i suoi demoni sugli altri nodi via ssh, e sui cluster
    # dove ssh dal login ai compute lo fa lo scheduler ma ssh FRA compute
    # no, la chiave dell'utente puo' semplicemente non essere autorizzata.
    # L'errore che ne segue (ORTE unable to start daemons / no route to
    # daemon) arriva dopo un minuto di attesa e non dice questo: meglio
    # scoprirlo qui in due secondi che nel log di mpirun.
    echo "=== connettivita' ssh fra i nodi ==="
    ssh_ok=1
    while read -r host; do
        if ssh -o BatchMode=yes -o ConnectTimeout=5 \
               -o StrictHostKeyChecking=accept-new \
               "$host" true 2> /dev/null; then
            echo "  $host: ok"
        else
            echo "  $host: FALLITA"
            ssh_ok=0
        fi
    done < <(sort -u "$PBS_NODEFILE")

    if [[ "$ssh_ok" -eq 0 ]]; then
        cat << 'RIMEDIO'

ATTENZIONE: ssh fra i nodi non funziona senza interazione, e mpirun gira
dentro un job dove non c'e' nessuno che possa digitare qualcosa.

La causa piu' comune non e' una chiave mancante ma una chiave CON
PASSPHRASE: se git te la chiede a ogni push, e' quella. Autorizzarla non
basta, resta indecifrabile senza terminale.

Serve una chiave dedicata senza passphrase. NON sovrascrivere quella che
usi per GitHub: creane una a parte e limitala ai nodi compute. La home e'
condivisa coi compute, quindi basta farlo una volta da login01.

  ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_cluster
  cat ~/.ssh/id_cluster.pub >> ~/.ssh/authorized_keys
  chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
  printf 'Host cpu* gpu*\n  IdentityFile ~/.ssh/id_cluster\n' >> ~/.ssh/config
  printf '  IdentitiesOnly yes\n  StrictHostKeyChecking accept-new\n' >> ~/.ssh/config
  chmod 600 ~/.ssh/config

RIMEDIO
        exit 1
    fi
else
    echo "PBS_NODEFILE non impostato: eseguo fuori da PBS, np=1 soltanto."
fi
echo

# ------------------------------------------------------------- compilazione

echo "=== compilazione ==="
core_sources=()
for source in src/*.c; do
    [[ "$(basename -- "$source")" == "main.c" ]] || core_sources+=("$source")
done

cflags=(-std=gnu11 -O3 -Wall -Wextra -Iinclude -mavx2 -DUSE_SIMD -DUSE_MPI
        -DDEFAULT_WIDTH="$NX" -DDEFAULT_HEIGHT="$NY" -DDEFAULT_DEPTH="$NZ"
        -DDEFAULT_T=1e-1 -DDEFAULT_STEPS="$STEPS")

printf '  mpi puro (senza OpenMP)   '
"${MPICC:-mpicc}" "${cflags[@]}" \
    test/paper_man.c "${core_sources[@]}" -lm -o "$build/mpi_only"
echo ok

printf '  ibrido (MPI + OpenMP)     '
"${MPICC:-mpicc}" "${cflags[@]}" -fopenmp -DUSE_OMP \
    test/paper_man.c "${core_sources[@]}" -fopenmp -lm -o "$build/hybrid"
echo ok

echo "  griglia ${NX}x${NY}x${NZ}, $STEPS passi"
echo

# ------------------------------------------------------------------ misura

# hostfile esplicito se PBS ce l'ha dato; altrimenti mpirun decide da solo
# (caso locale, np=1).
mpirun_hosts=()
[[ -n "${PBS_NODEFILE:-}" ]] && mpirun_hosts=(--hostfile "$PBS_NODEFILE")

# Un caso, un file con l'output grezzo e le quattro righe di errore estratte.
# --bind-to none e' la stessa cautela di run_thread_scaling.sh: con piu' di
# un rank mpirun pinnerebbe ognuno a un core solo, lasciando ai suoi thread
# OpenMP quell'unico core da spartirsi invece di uno a testa.
#
# mpiopts e progargs sono stringhe non quotate all'uso apposta: servono a
# separare le opzioni di mpirun (prima dell'eseguibile) dagli argomenti del
# programma (dopo), e qui sono sempre costanti scritte da questo script, mai
# input esterno.
run_case()
{
    local nome="$1" exe="$2" np="$3" threads="$4" mpiopts="$5" progargs="$6"
    local out_file="$build/$nome.out"

    OMP_NUM_THREADS="$threads" OMP_PROC_BIND=close OMP_PLACES=cores \
        mpirun --bind-to none "${mpirun_hosts[@]}" -n "$np" $mpiopts \
        "$exe" $progargs > "$out_file" 2>&1

    grep '^  L2 error' "$out_file" > "$build/$nome.norms"
    printf '  %-28s ' "$nome"
    if [[ -s "$build/$nome.norms" ]]; then
        echo ok
    else
        echo "NESSUN OUTPUT ATTESO -- vedi $out_file"
    fi
}

echo "=== riferimento: 1 rank, nessuna comunicazione ==="
run_case riferimento "$build/mpi_only" 1 1 "" ""

echo
echo "=== multi-nodo: 4 rank, un nodo a testa, forma 1x2x2 ==="
run_case multinodo_1 "$build/mpi_only" 4 1 "--map-by node" "1 2 2"
run_case multinodo_2 "$build/mpi_only" 4 1 "--map-by node" "1 2 2"

if [[ "${n_unique:-1}" -ge 2 ]]; then
    echo
    echo "=== multi-nodo ibrido: 4 rank x 7 thread, un nodo a testa ==="
    run_case ibrido "$build/hybrid" 4 7 "--map-by node" "1 2 2"
fi

# ------------------------------------------------------------------ verdetto

echo
echo "=== verdetto ==="

confronta()
{
    local a="$1" b="$2" etichetta="$3"
    if [[ ! -f "$build/$a.norms" || ! -f "$build/$b.norms" ]]; then
        printf '  %-40s SALTATO (manca un file)\n' "$etichetta"
        return
    fi
    if diff -q "$build/$a.norms" "$build/$b.norms" > /dev/null; then
        printf '  %-40s OK, identiche a 10 cifre\n' "$etichetta"
    else
        printf '  %-40s DIFFERENTI:\n' "$etichetta"
        diff "$build/$a.norms" "$build/$b.norms" | sed 's/^/      /'
        FAILED=1
    fi
}

FAILED=0
confronta multinodo_1 multinodo_2 "multi-nodo, due ripetizioni fra loro"
confronta riferimento multinodo_1 "multi-nodo contro il riferimento"
[[ -f "$build/ibrido.norms" ]] && \
    confronta riferimento ibrido "ibrido multi-nodo contro il riferimento"

echo
if [[ "$FAILED" -eq 0 ]]; then
    echo "L'halo attraverso la rete da' lo stesso risultato della RAM condivisa."
else
    echo "ATTENZIONE: differenze trovate. Vedi sopra e i file in $build/*.out"
fi

echo
echo "output grezzo e norme in $build/"
echo "log in $log"

exit "$FAILED"
