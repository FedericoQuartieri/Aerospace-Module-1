#!/usr/bin/env bash
#PBS -N nsb-multinode-debug
#PBS -q cpu
#PBS -l select=2:ncpus=2
#PBS -l place=scatter
#PBS -l walltime=00:15:00
#PBS -j oe
#
# Perche' non parte niente su piu' nodi, e quale strada invece funziona.
#
#   qsub scripts/run_multinode_debug.sh       sul cluster
#   ./scripts/run_multinode_debug.sh          in locale (prova a vuoto)
#
# ----------------------------------------------------------------------------
# Cosa fa e cosa NON fa
# ----------------------------------------------------------------------------
#
# Non risolve niente e non misura niente: prova a lanciare due processi su due
# nodi in nove modi diversi e per ognuno registra se e' partito e, se no, con
# che errore esatto. Serve a smettere di tirare a indovinare, e a poter aprire
# un ticket agli amministratori con una riga sola invece che con "MPI non
# funziona".
#
# Chiede due chunk da 2 CPU: quattro CPU in tutto, ben sotto il tetto di 28
# della coda `cpu', e place=scatter obbliga un chunk per host. Quindici minuti
# bastano perche' ogni prova ha un timeout: un launcher rotto tipicamente
# resta appeso a lungo, e senza timeout la prima strada che fallisce si
# mangerebbe il walltime prima di arrivare alle altre.
#
# ----------------------------------------------------------------------------
# Le tre strade gia' escluse, e le due mai provate
# ----------------------------------------------------------------------------
#
# MULTITHREAD.md 9.3 riporta tre esclusioni:
#
#   plm tm       il componente non c'e' nell'Open MPI di sistema
#   ssh          sshd sui compute non accetta la chiave dell'utente
#   pbs_tmrsh    risponde `tm_poll(obit): err not connected'
#
# Restano fuori due cose che nessuna delle tre copre:
#
#   --prefix     e' LA causa piu' comune di "unable to start daemons" quando
#                ssh in se' funziona. mpirun si collega al nodo remoto con una
#                shell non interattiva, che non carica i moduli: PATH non
#                contiene `orted', mpirun aspetta un demone che non partira'
#                mai, e l'errore che stampa parla di demoni, non di PATH.
#                Si aggira dicendogli dove sta l'installazione.
#                Se l'errore osservato era quello, questa e' la spiegazione,
#                e non e' un problema di chiavi.
#
#   Hydra        MPICH e Intel MPI non usano il plm di Open MPI ma Hydra, che
#                ha un launcher PBS suo che parla TM direttamente. Funziona
#                regolarmente dove manca `plm tm', perche' e' un binario
#                diverso compilato da un'altra parte. Se sul cluster c'e' un
#                modulo mpich o intel-mpi, questa e' la strada piu' probabile
#                fra tutte.
#
# ----------------------------------------------------------------------------
# Perche' pbsdsh viene prima di tutto
# ----------------------------------------------------------------------------
#
# pbsdsh e' il lanciatore nativo di PBS: parla TM senza passare da MPI ne' da
# ssh. E' il test acido, e divide il problema in due meta' che vogliono
# risposte opposte:
#
#   pbsdsh funziona    TM va, i nodi sono raggiungibili, e il problema e'
#                      solo in COME l'MPI installato prova a usarli. E' un
#                      problema risolvibile da questo lato.
#
#   pbsdsh fallisce    il sito ha TM ristretto o rotto. Nessun launcher MPI
#                      basato su TM potra' funzionare, e l'unica strada resta
#                      ssh. E' un ticket, non una configurazione.
#
# In piu', se funziona, pbsdsh diventa il veicolo per ispezionare l'altro nodo
# senza ssh: e' cosi' che le prove qui sotto verificano se la home e' condivisa
# e se orted e' nel PATH remoto, che sono le due domande a cui l'errore di
# mpirun non risponde mai.

set -uo pipefail        # niente -e: qui fallire e' il dato, non un incidente

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

build="build/multinode-debug"
log="$build/run.log"
mkdir -p "$build"
exec > >(tee "$log") 2>&1

TIMEOUT="${TIMEOUT:-75}"
esiti=""            # "nome|esito|dettaglio" per riga, per il riassunto finale

annota() { esiti="$esiti$1|$2|$3"$'\n'; }

# ---------------------------------------------------------------- 1. contesto

echo "=============================================================="
echo " 1. contesto"
echo "=============================================================="
printf 'host di lancio:   %s\n' "$(hostname)"
printf 'PBS_JOBID:        %s\n' "${PBS_JOBID:-(non impostato)}"
printf 'PBS_ENVIRONMENT:  %s\n' "${PBS_ENVIRONMENT:-(non impostato)}"
printf 'PBS_NODEFILE:     %s\n' "${PBS_NODEFILE:-(non impostato)}"

nodi=()
if [[ -n "${PBS_NODEFILE:-}" && -f "$PBS_NODEFILE" ]]; then
    mapfile -t nodi < <(sort -u "$PBS_NODEFILE")
    printf 'nodi distinti:    %s  (%s)\n' "${#nodi[@]}" "${nodi[*]}"
    if [[ "${#nodi[@]}" -lt 2 ]]; then
        echo
        echo "ATTENZIONE: un nodo solo. Senza place=scatter lo scheduler puo'"
        echo "impacchettare i chunk sullo stesso host, e allora non c'e'"
        echo "niente da diagnosticare: le prove passerebbero tutte."
        echo "Rilancia con:  qsub -l select=2:ncpus=2 -l place=scatter"
    fi
else
    echo
    echo "Fuori da PBS: le prove che richiedono TM verranno saltate e quelle"
    echo "su ssh useranno localhost. Serve a vedere che lo script gira."
fi
altro="${nodi[1]:-}"
echo

# --------------------------------------------------------------- 2. cosa c'e'

echo "=============================================================="
echo " 2. che MPI c'e' installato"
echo "=============================================================="
for strumento in mpirun mpiexec mpiexec.hydra pbsdsh pbs_tmrsh orted ompi_info; do
    printf '  %-14s %s\n' "$strumento" "$(command -v "$strumento" || echo '(assente)')"
done
echo
if command -v mpirun > /dev/null; then
    printf 'versione:  %s\n' "$(mpirun --version 2>&1 | head -1)"
fi

# I componenti plm sono la lista dei modi in cui Open MPI sa avviare demoni
# remoti. Se `tm' non c'e', mpirun non puo' parlare con PBS, punto: qualunque
# `-mca plm tm' verra' ignorato in silenzio e si ricadra' su rsh.
if command -v ompi_info > /dev/null; then
    echo
    echo "componenti plm di Open MPI (i modi in cui sa avviare demoni remoti):"
    ompi_info --parsable 2>/dev/null | awk -F: '/^mca:plm:/ {print $3}' |
        sort -u | sed 's/^/  /'
fi

# I moduli sono la strada per un MPI diverso da quello di sistema, ed e'
# esattamente quello che serve se il plm tm manca.
echo
echo "moduli MPI disponibili:"
if [[ -r /etc/profile.d/modules.sh ]]; then
    # `module' e' una funzione di shell: in uno script non interattivo va
    # caricata a mano, altrimenti "command not found" e si conclude a torto
    # che i moduli non ci sono.
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh 2> /dev/null
fi
if command -v module > /dev/null 2>&1; then
    module avail 2>&1 | grep -iE 'mpi|mpich|openmpi|hpcx|impi|intel' |
        sed 's/^/  /' | head -30
    [[ "${PIPESTATUS[1]}" -eq 0 ]] || echo "  (nessun modulo con 'mpi' nel nome)"
else
    echo '  (comando module non disponibile: MPI alternativi non verificabili)'
fi
echo

# ----------------------------------------------------------- 3. il programma

echo "=============================================================="
echo " 3. programma di prova"
echo "=============================================================="
cat > "$build/hello.c" <<'HELLO'
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    char host[256];
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(host, sizeof host);
    printf("rank %d/%d su %s\n", rank, size, host);
    MPI_Finalize();
    return 0;
}
HELLO

if "${MPICC:-mpicc}" "$build/hello.c" -o "$build/hello" 2> "$build/compile.err"; then
    echo "  compilato: $build/hello"
else
    echo "  COMPILAZIONE FALLITA:"
    sed 's/^/    /' "$build/compile.err"
    exit 1
fi
echo

# Ogni prova gira con un timeout e finisce in un file suo. Il criterio di
# successo non e' il codice di uscita -- un launcher puo' uscire a zero senza
# aver messo piede sull'altro nodo -- ma quanti host DIVERSI si sono
# presentati nell'output.
prova()
{
    local nome="$1" attesi="$2"
    shift 2
    local out="$build/$nome.out"

    printf '  %-34s ' "$nome"
    timeout "$TIMEOUT" "$@" > "$out" 2>&1
    local rc=$?

    local host_distinti
    host_distinti="$(grep -oP '(?<= su )\S+' "$out" 2> /dev/null | sort -u | wc -l)"

    if [[ "$rc" -eq 124 ]]; then
        echo "APPESO (timeout ${TIMEOUT}s)"
        annota "$nome" "appeso" "nessuna risposta in ${TIMEOUT}s"
    elif [[ "$host_distinti" -ge "$attesi" ]]; then
        echo "OK  ($host_distinti host distinti)"
        annota "$nome" "OK" "$host_distinti host"
    elif [[ "$host_distinti" -ge 1 ]]; then
        echo "PARZIALE (tutto su $host_distinti host)"
        annota "$nome" "parziale" "girato su $host_distinti host solo"
    else
        local motivo
        motivo="$(grep -m1 -iE 'error|denied|not found|unable|refused|no route|failed|err ' "$out" |
                  cut -c1-72)"
        echo "FALLITO"
        [[ -n "$motivo" ]] && printf '%38s%s\n' '' "-> $motivo"
        annota "$nome" "fallito" "${motivo:-nessun messaggio, vedi $out}"
    fi
}

# ------------------------------------------------------ 4. TM, senza MPI

echo "=============================================================="
echo " 4. TM nudo: pbsdsh (il test acido)"
echo "=============================================================="
if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "  saltato: fuori da un job PBS"
    annota "pbsdsh" "saltato" "fuori da PBS"
elif ! command -v pbsdsh > /dev/null; then
    echo "  pbsdsh non installato"
    annota "pbsdsh" "assente" "binario non trovato"
else
    printf '  %-34s ' "pbsdsh -- hostname"
    if timeout 30 pbsdsh -- /bin/hostname > "$build/pbsdsh.out" 2>&1; then
        distinti="$(sort -u "$build/pbsdsh.out" | grep -c . )"
        echo "OK ($distinti host)"
        sed 's/^/      /' "$build/pbsdsh.out" | head -5
        annota "pbsdsh" "OK" "$distinti host -- TM funziona"
        tm_ok=1
    else
        echo "FALLITO"
        sed 's/^/      /' "$build/pbsdsh.out" | head -5
        annota "pbsdsh" "fallito" "$(head -1 "$build/pbsdsh.out" | cut -c1-60)"
        tm_ok=0
    fi

    # Se TM va, si puo' guardare dentro l'altro nodo senza ssh: sono le due
    # domande a cui l'errore di mpirun non risponde mai.
    if [[ "${tm_ok:-0}" -eq 1 ]]; then
        echo
        echo "  ispezione dell'altro nodo attraverso TM:"
        printf '    home condivisa    '
        timeout 30 pbsdsh -n 1 -- /bin/ls "$HOME/.ssh/authorized_keys" \
            > "$build/tm_home.out" 2>&1 && echo "si" || {
                echo "NO -- authorized_keys non si vede dall'altro nodo"
                sed 's/^/      /' "$build/tm_home.out" | head -3; }
        printf '    orted nel PATH    '
        timeout 30 pbsdsh -n 1 -- /bin/sh -c 'command -v orted' \
            > "$build/tm_orted.out" 2>&1 && sed -n '1p' "$build/tm_orted.out" || {
                echo "NO -- ecco perche' mpirun aspetta demoni che non partono"
                echo "                      (e' il caso che --prefix risolve)"; }
    fi
fi
echo

# ------------------------------------------------------------------- 5. ssh

echo "=============================================================="
echo " 5. ssh verso l'altro nodo, un metodo per volta"
echo "=============================================================="
bersaglio="${altro:-localhost}"
echo "  bersaglio: $bersaglio"
echo

# Separare i metodi conta: "Permission denied (publickey)" non dice se la
# chiave e' stata rifiutata o se non e' mai stata offerta, e hostbased --
# il meccanismo classico dei cluster, via shosts.equiv -- e' spesso l'unico
# abilitato dal sito e nessuno lo prova mai.
for metodo in publickey hostbased gssapi-with-mic keyboard-interactive; do
    printf '  %-24s ' "$metodo"
    if timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 \
            -o StrictHostKeyChecking=accept-new \
            -o "PreferredAuthentications=$metodo" \
            "$bersaglio" hostname > "$build/ssh_$metodo.out" 2>&1; then
        echo "OK -> $(cat "$build/ssh_$metodo.out")"
        annota "ssh:$metodo" "OK" "funziona"
    else
        echo "no"
        annota "ssh:$metodo" "fallito" "$(grep -m1 -i 'denied\|refused\|timed out' \
            "$build/ssh_$metodo.out" | cut -c1-60)"
    fi
done

echo
echo "  motivo esatto lato client (ssh -vvv, righe che contano):"
timeout 20 ssh -vvv -o BatchMode=yes -o ConnectTimeout=8 \
    -o StrictHostKeyChecking=accept-new \
    "$bersaglio" hostname > "$build/ssh_verbose.out" 2>&1
grep -iE 'Authentications that can continue|Offering public key|Server accepts key|Trying private key|No more authentication methods|Permission denied|Connection closed|banner' \
    "$build/ssh_verbose.out" | sed 's/^/    /' | head -12
echo "  (completo in $build/ssh_verbose.out)"
echo

# --------------------------------------------------------- 6. i launcher

echo "=============================================================="
echo " 6. lanciare due processi su due nodi"
echo "=============================================================="

if [[ "${#nodi[@]}" -lt 2 ]]; then
    echo "  meno di due nodi: le prove direbbero OK senza provare niente."
    echo "  Saltate."
else
    hostfile=(--hostfile "$PBS_NODEFILE")
    exe="$build/hello"

    echo
    echo "  --- Open MPI, come lo si lancia di solito ---"
    prova "ompi_default" 2 \
        mpirun "${hostfile[@]}" --map-by node -n 2 "$exe"

    prova "ompi_plm_tm" 2 \
        mpirun -mca plm tm -n 2 "$exe"

    prova "ompi_rsh_ssh" 2 \
        mpirun -mca plm rsh -mca plm_rsh_agent ssh \
            "${hostfile[@]}" --map-by node -n 2 "$exe"

    prova "ompi_rsh_pbs_tmrsh" 2 \
        mpirun -mca plm rsh -mca plm_rsh_agent pbs_tmrsh \
            "${hostfile[@]}" --map-by node -n 2 "$exe"

    # La strada mai provata: dire a mpirun dove sta l'installazione, cosi' il
    # nodo remoto trova orted anche con una shell che non carica i moduli.
    echo
    echo "  --- la stessa cosa, dicendo dove sta orted (--prefix) ---"
    prefix=""
    if command -v mpirun > /dev/null; then
        prefix="$(dirname "$(dirname "$(readlink -f "$(command -v mpirun)")")")"
        echo "      prefix dedotto: $prefix"
    fi
    if [[ -n "$prefix" ]]; then
        prova "ompi_ssh_prefix" 2 \
            mpirun --prefix "$prefix" -mca plm rsh -mca plm_rsh_agent ssh \
                "${hostfile[@]}" --map-by node -n 2 "$exe"

        prova "ompi_tmrsh_prefix" 2 \
            mpirun --prefix "$prefix" -mca plm rsh -mca plm_rsh_agent pbs_tmrsh \
                "${hostfile[@]}" --map-by node -n 2 "$exe"
    fi

    # L'altra strada mai provata: un MPI che non usa il plm di Open MPI.
    echo
    echo "  --- Hydra (MPICH / Intel MPI): launcher PBS proprio ---"
    hydra="$(command -v mpiexec.hydra || true)"
    if [[ -z "$hydra" ]] && mpiexec -help 2>&1 | grep -qi 'launcher'; then
        hydra="$(command -v mpiexec)"
    fi
    if [[ -z "$hydra" ]]; then
        echo "      Hydra non presente nell'ambiente attuale."
        echo "      Se al punto 2 compare un modulo mpich o intel-mpi, questa"
        echo "      e' la prova che vale la pena rifare dopo averlo caricato:"
        echo "        module load <modulo>"
        echo "        mpicc build/multinode-debug/hello.c -o /tmp/hello"
        echo "        mpiexec -launcher pbs -n 2 /tmp/hello"
        annota "hydra" "assente" "nessun mpiexec.hydra nell'ambiente"
    else
        echo "      hydra: $hydra"
        prova "hydra_launcher_pbs" 2 \
            "$hydra" -launcher pbs -n 2 "$exe"
        prova "hydra_launcher_ssh" 2 \
            "$hydra" -launcher ssh -f "$PBS_NODEFILE" -n 2 "$exe"
        prova "hydra_tmrsh" 2 \
            "$hydra" -launcher rsh -launcher-exec "$(command -v pbs_tmrsh || echo pbs_tmrsh)" \
                -f "$PBS_NODEFILE" -n 2 "$exe"
    fi
fi
echo

# ---------------------------------------------------------------- 7. verdetto

echo "=============================================================="
echo " 7. riassunto"
echo "=============================================================="
echo
printf '  %-24s %-10s %s\n' PROVA ESITO DETTAGLIO
printf '  %-24s %-10s %s\n' "------------------------" "----------" "--------------------"
printf '%s' "$esiti" | while IFS='|' read -r nome esito dettaglio; do
    [[ -n "$nome" ]] && printf '  %-24s %-10s %s\n' "$nome" "$esito" "$dettaglio"
done

echo
vincenti="$(printf '%s' "$esiti" | awk -F'|' '$2 == "OK" && $1 !~ /^ssh:|^pbsdsh/ {print $1}')"
if [[ -n "$vincenti" ]]; then
    echo "  ALMENO UNA STRADA FUNZIONA:"
    sed 's/^/    /' <<< "$vincenti"
    echo
    echo "  Prendi la riga corrispondente da questo script e mettila in"
    echo "  run_multinode_check.sh al posto del suo mpirun."
else
    if [[ "${#nodi[@]}" -lt 2 ]]; then
        # Senza due nodi le prove del punto 6 non sono state eseguite affatto:
        # concludere qualcosa qui vorrebbe dire scambiare "non provato" per
        # "non funziona", che e' l'errore che questo script esiste per evitare.
        echo "  INCONCLUSIVO: il punto 6 non e' stato eseguito, perche' il job"
        echo "  non aveva due nodi. Quanto sopra dice solo cosa c'e' installato."
        echo
        echo "  Serve sottometterlo davvero:"
        echo "    qsub scripts/run_multinode_debug.sh"
        exit 0
    fi

    echo "  Nessun launcher e' riuscito ad arrivare su due nodi."
    echo
    if printf '%s' "$esiti" | grep -q '^pbsdsh|OK'; then
        cat << 'TICKET'
  Ma pbsdsh SI: TM funziona e i nodi sono raggiungibili. Il problema e'
  solo che nessun MPI installato sa usarlo. E' un ticket con una richiesta
  precisa, che e' molto piu' facile da soddisfare di "MPI non funziona":

    "L'Open MPI di sistema (4.1.6) e' compilato senza supporto TM: manca il
     componente plm tm, e restano solo rsh/slurm/isolated. Su un cluster PBS
     senza ssh fra i nodi di calcolo questo rende impossibile lanciare job
     MPI multi-nodo. pbsdsh funziona, quindi TM e' attivo.
     Chiedo una di queste due cose:
       - un Open MPI ricompilato con --with-tm, oppure
       - un modulo MPICH o Intel MPI, il cui launcher Hydra parla TM
         direttamente (mpiexec -launcher pbs)."
TICKET
    else
        cat << 'TICKET'
  E nemmeno pbsdsh: TM stesso non risponde (vedi il punto 4). Questo e' lato sito, non
  configurabile dall'utente. Il ticket e':

    "pbsdsh dentro un job con select=2:ncpus=2 -l place=scatter non riesce
     a eseguire sul secondo chunk. Anche pbs_tmrsh risponde
     `tm_poll(obit): err not connected'. Sembra che l'interfaccia TM non
     sia utilizzabile dai job utente: e' voluto? In tal caso, qual e' il
     modo supportato di lanciare un job MPI su piu' nodi su questo cluster?"
TICKET
    fi
fi

echo
echo "  output completo di ogni prova in $build/"
echo "  log in $log"
