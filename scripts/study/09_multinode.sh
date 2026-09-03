#!/usr/bin/env bash
#PBS -N nsb-09-multinode
#PBS -q cpu
#PBS -l select=4:ncpus=7
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe
#
# Fase 9 -- fuori dal nodo: la domanda che il cluster non lasciava fare.
#
# Stato noto, e verificato in tre modi indipendenti (MULTITHREAD.md §9.3):
# su questo sito non esiste un modo funzionante di lanciare processi su piu'
# nodi. L'Open MPI di sistema non ha il componente `plm tm' (solo rsh, slurm,
# isolated), sshd sui compute non accetta la chiave dell'utente, e
# `pbs_tmrsh' risponde `tm_poll(obit): err not connected'.
#
# Questa fase non da' per scontato che sia ancora cosi'. Prima prova le tre
# strade, e le prova in fretta; poi, se una funziona, misura -- perche' e'
# proprio fuori dal nodo che la domanda schur/pipeline diventa interessante:
#
#   schur    paga in aritmetica (tre risoluzioni locali per linea) e in
#            collettive. Le collettive su rete costano di piu', ma sono poche
#            e grosse.
#   pipeline paga in latenza, e basta: la sua efficienza e' 3B/(3B+P-1) con
#            un messaggio ogni batch. Su RAM condivisa la latenza e' bassa e
#            la pipeline puo' permetterselo; su rete e' l'ordine di grandezza
#            che cambia, e la formula di PIPELINE.md §1.6 dice che il batch
#            ottimo cresce come sqrt(1/lambda). E' l'unico posto dove quella
#            formula si puo' falsificare.
#
# La coda e la select non sono le stesse delle altre fasi: la coda `cpu'
# concede al massimo 28 CPU per job sommate su tutti i chunk (select=2:ncpus=28
# viene rifiutato, select=4:ncpus=7 no), e i suoi nodi non sono esclusivi.
# Quindi i tempi di questa fase sono indicativi e nel CSV portano una nota che
# lo dice; la parte che NON dipende dai vicini -- se il risultato resta lo
# stesso attraversando la rete -- vale comunque.
#
#   qsub scripts/study/09_multinode.sh

source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"

cd "${PBS_O_WORKDIR:-$STUDY_ROOT}"

# La 09 non si ri-sottomette da sola: o le tre vie sono aperte e misura in
# pochi minuti, o sono chiuse ed esce subito. Ripetere non cambierebbe niente.
STUDY_CHAINABLE=0
STUDY_BUDGET="${STUDY_BUDGET:-3000}"

study_begin 09_multinode
study_machine

GRID="${GRID:-96 96 96}"
STEPS="${STEPS:-10}"

# ---------------------------------------------------------------- le tre vie

echo "=== nodi assegnati ==="
nodes=1
if [[ -n "${PBS_NODEFILE:-}" && -f "$PBS_NODEFILE" ]]; then
    sort -u "$PBS_NODEFILE" | sed 's/^/  /'
    nodes="$(sort -u "$PBS_NODEFILE" | wc -l)"
fi
echo "  nodi distinti: $nodes"
echo

if [[ "$nodes" -lt 2 ]]; then
    echo "  Un nodo solo: questa fase non ha niente da misurare che le altre"
    echo "  non misurino gia'. Serve  qsub -l select=4:ncpus=7 -l place=scatter"
    study_end
    exit 0
fi

echo "=== via 1: componenti di lancio disponibili ==="
if command -v ompi_info > /dev/null; then
    plm="$(ompi_info --param plm all 2>/dev/null | awk -F: '/MCA plm:/ {print $3}' | tr -d ' ' | sort -u | paste -sd, -)"
    echo "  plm: $plm"
    case "$plm" in
        *tm*) printf '  plm tm c\x27e\x27: PBS puo\x27 avviare i processi da solo.\n' ;;
        *)    printf '  plm tm assente: resta ssh o pbs_tmrsh.\n' ;;
    esac
else
    echo "  ompi_info non disponibile"
fi
echo

echo "=== via 2: ssh diretto fra i nodi ==="
ssh_ok=1
while read -r host; do
    if ssh -o BatchMode=yes -o ConnectTimeout=5 \
           -o StrictHostKeyChecking=accept-new "$host" true 2> /dev/null; then
        echo "  $host: ok"
    else
        echo "  $host: FALLITA"
        ssh_ok=0
    fi
done < <(sort -u "$PBS_NODEFILE")
echo

echo "=== via 3: pbs_tmrsh ==="
tm_ok=0
if command -v pbs_tmrsh > /dev/null; then
    other="$(sort -u "$PBS_NODEFILE" | grep -v "^$(hostname)$" | head -1)"
    if [[ -n "$other" ]] && timeout 30 pbs_tmrsh "$other" true 2>&1 | tee "$STUDY_OUT/tmrsh.log"; then
        echo "  pbs_tmrsh verso $other: ok"
        tm_ok=1
    else
        echo "  pbs_tmrsh verso $other: fallito ($(head -1 "$STUDY_OUT/tmrsh.log" 2>/dev/null))"
    fi
else
    echo "  pbs_tmrsh non presente"
fi
echo

# La prova che conta: un mpirun banale che deve solo stampare i nomi degli host.
echo "=== prova di lancio ==="
launch_ok=0
if timeout 120 "${MPIRUN:-mpirun}" --hostfile "$PBS_NODEFILE" --map-by node \
        -n "$nodes" hostname > "$STUDY_OUT/launch.log" 2>&1; then
    sort -u "$STUDY_OUT/launch.log" | sed 's/^/  /'
    launch_ok=1
    echo "  il lancio multi-nodo funziona"
else
    echo "  fallito:"
    sed 's/^/    /' "$STUDY_OUT/launch.log" | head -12
fi
echo

if [[ "$launch_ok" -eq 0 ]]; then
    cat << 'NOTA'
  Il multi-nodo resta bloccato dall'infrastruttura, non dal codice. Le tre
  vie sono quelle di §9.3 e vanno sbloccate da chi amministra il cluster:
  o `plm tm' compilato dentro Open MPI, o sshd che accetti la chiave
  dell'utente sui compute, o pbs_tmrsh funzionante.

  Tutto il resto dello studio riguarda un nodo solo, e va letto sapendolo:
  la pipeline paga in latenza, e su RAM condivisa la latenza e' la piu'
  bassa che vedra' mai.
NOTA
    study_end
    exit 0
fi

# ------------------------------------------------------------------ misura

echo "=== attraversando la rete ==="
echo "    (coda 'cpu', nodi non esclusivi: i tempi sono indicativi, le norme no)"

# Il piazzamento di questa fase non e' quello delle altre: un rank per nodo,
# scelto dall'hostfile che PBS ha scritto.
export PLACEMENT_OVERRIDE="--hostfile $PBS_NODEFILE --map-by node --bind-to none"
CASE_GRID="$GRID"
CASE_STEPS="$STEPS"
CASE_SIMD=1
CASE_OMP=0
CASE_MPI=1
CASE_THREADS=1
CASE_NORMS=1
CASE_REPEATS=2
CASE_TIMEOUT="${CASE_TIMEOUT:-900}"
CASE_NOTE="nodi non esclusivi"

# Riferimento sullo stesso job: un rank, nessuna rete.
for backend in schur pipeline; do
    study_case label="$backend 1 rank" backend="$backend" ranks=1 \
        shape="1 1 1" note="riferimento, nessuna rete"
done

# Un rank per nodo, forma 1x2x2: tutti e tre gli assi divisi, quindi tutte e
# tre le direzioni di halo passano dalla rete almeno una volta.
for backend in schur pipeline; do
    study_case label="$backend ${nodes} nodi" backend="$backend" \
        ranks="$nodes" shape="1 2 2" note="un rank per nodo; $CASE_NOTE"
done

# Il batch della pipeline, dove la latenza e' quella vera: e' l'unico posto in
# cui B_ottimo ~ sqrt((P-1)W/(S*lambda)) puo' essere smentito.
for batch in 16 64 256 1024; do
    study_case label="pipeline rete batch=$batch" backend=pipeline \
        batch="$batch" ranks="$nodes" shape="1 2 2" \
        note="un rank per nodo; $CASE_NOTE"
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== verdetto ==="
    awk -F, -v phase=09_multinode '
    NR == 1 || $1 != phase || $32 != "ok" { next }
    {
        printf "  %-26s %10.1f ms   L2 u_x = %s\n", $2, $17, $30
        if ($30 != "") { norms[$30 "|" $31] = 1; n++ }
    }
    END {
        print ""
        k = 0
        for (key in norms) k++
        if (k == 1)
            print "  Tutte le configurazioni, rete compresa, danno le stesse norme."
        else
            printf "  ATTENZIONE: %d risposte diverse su %d misure. Lo scambio\n  attraverso la rete non e\x27 equivalente a quello in RAM.\n", k, n
    }' "$STUDY_CSV"
fi

study_end
