# shellcheck shell=bash
#
# Le parti comuni alle fasi dello studio di scaling.
#
# Ogni fase e' uno script indipendente, sottomettibile da solo con qsub, che
# fa una domanda sola. Quello che condividono sta qui: come si descrive la
# macchina, come si compila una variante, come si esegue un caso e come lo si
# scrive nel CSV. Una riga di CSV ha sempre le stesse 33 colonne, qualunque
# fase l'abbia prodotta, cosi' i risultati si concatenano e si confrontano
# senza adattatori.
#
# Tre scelte che valgono per tutte le fasi:
#
#   ripresa      il CSV si scrive una riga alla volta e ogni caso concluso
#                lascia la sua chiave in done.keys. Se il walltime uccide il
#                job, ri-sottometterlo riparte da dove era arrivato invece di
#                rifare tutto. FRESH=1 ricomincia da capo.
#
#   timeout      ogni caso ha un tetto di tempo. Una configurazione che va in
#                stallo -- e nella zona mista qualcuna ci va vicino -- non si
#                porta via il resto del job: lascia una riga con status
#                `timeout` e si prosegue.
#
#   binari in cache   compilare e' l'unica cosa che non si vuole ripetere: una
#                variante (backend, simd, omp, mpi, batch) si costruisce una
#                volta e resta in build/study/bin.
#
#   budget       la coda `scalability' concede 30 minuti per job
#                (resources_max.walltime = 00:30:00) e lo studio ne vuole molte
#                di piu'. Ogni fase lavora quindi a budget: quando il tempo
#                utile e' finito smette PRIMA di essere uccisa -- cosi' il caso
#                in corso non viene troncato a meta' -- e si ri-sottomette da
#                sola per continuare. Un walltime scaduto non perde niente e
#                non richiede nessun intervento.

set -euo pipefail

STUDY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
STUDY_BASE="$STUDY_ROOT/build/study"
STUDY_BIN="$STUDY_BASE/bin"

# Colonne, una volta sola: il resto del file si riferisce a questa riga.
STUDY_HEADER='phase,label,backend,batch,simd,omp,mpi,ranks,threads,nx,ny,nz,steps,px,py,pz,wall_ms,mpi_ms,eta_ms,zeta_ms,u_ms,psi_ms,philow_ms,phihigh_ms,pressure_ms,porosity_ms,untimed_ms,cellstep_1e8s,rss_mb,l2_ux,l2_p,status,note'

# Quanti campi produce l'awk di lettura: serve a riempire di vuoti la riga di
# un caso fallito senza sfasare le colonne.
STUDY_MEASURED_FIELDS=18

# ---------------------------------------------------------------- avvio fase

study_begin()
{
    STUDY_PHASE="$1"
    STUDY_OUT="$STUDY_BASE/$STUDY_PHASE"
    STUDY_CSV="$STUDY_OUT/results.csv"
    STUDY_KEYS="$STUDY_OUT/done.keys"
    STUDY_LOG="$STUDY_OUT/run.log"
    STUDY_STARTED="$(date +%s)"
    STUDY_CASES=0
    STUDY_SKIPPED=0
    STUDY_FAILED=0
    STUDY_PENDING=0
    STUDY_OUT_OF_TIME=0
    # Lavoro utile per job. Il resto del walltime serve alla compilazione, al
    # riepilogo e al margine per chiudere il caso in corso senza essere uccisi
    # nel mezzo: una misura troncata dal walltime non finisce nel CSV, e il
    # tempo speso a produrla e' perso.
    STUDY_BUDGET="${STUDY_BUDGET:-1500}"
    STUDY_CHAIN="${STUDY_CHAIN:-1}"
    STUDY_CHAIN_MAX="${STUDY_CHAIN_MAX:-80}"
    # Le fasi che si ri-sottomettono da sole. La 00 e la 09 no: la prima
    # perche' le altre dipendono dalla sua riuscita e devono partire quando ha
    # finito davvero, la seconda perche' o funziona in un minuto o non
    # funziona affatto.
    STUDY_CHAINABLE="${STUDY_CHAINABLE:-1}"

    mkdir -p "$STUDY_OUT" "$STUDY_BIN"

    if [[ "${FRESH:-0}" == "1" ]]; then
        rm -f "$STUDY_CSV" "$STUDY_KEYS"
    fi
    [[ -f "$STUDY_CSV" ]] || printf '%s\n' "$STUDY_HEADER" > "$STUDY_CSV"
    touch "$STUDY_KEYS"

    # Su questo sito il .o<jobid> di PBS non arriva ne' nella directory di
    # sottomissione ne' nella home: il log va tenuto accanto al CSV, e in
    # append perche' una ripresa non deve cancellare quello di prima.
    exec > >(tee -a "$STUDY_LOG") 2>&1

    echo "==============================================================="
    echo "fase $STUDY_PHASE -- $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==============================================================="
    echo "risultati: $STUDY_CSV"
    printf 'budget:    %d min di lavoro utile (job %d della catena)\n' \
        $(( STUDY_BUDGET / 60 )) "$STUDY_CHAIN"
    [[ "${DRY_RUN:-0}" == "1" ]] && echo "DRY_RUN=1: elenco i casi, non eseguo"
    echo
}

study_end()
{
    local elapsed=$(( $(date +%s) - STUDY_STARTED ))

    echo
    echo "=== fine fase $STUDY_PHASE ==="
    printf 'casi eseguiti: %d   saltati (gia\x27 fatti): %d   falliti: %d\n' \
        "$STUDY_CASES" "$STUDY_SKIPPED" "$STUDY_FAILED"
    printf 'tempo totale:  %02d:%02d:%02d\n' \
        $(( elapsed / 3600 )) $(( elapsed % 3600 / 60 )) $(( elapsed % 60 ))
    echo "csv: $STUDY_CSV"
    echo "log: $STUDY_LOG"

    if [[ "$STUDY_OUT_OF_TIME" -eq 0 ]]; then
        [[ "${DRY_RUN:-0}" == "1" ]] || echo "la fase e' completa."
        return 0
    fi

    printf 'restano %d casi: il budget di questo job e\x27 finito.\n' \
        "$STUDY_PENDING"
    study_resubmit
}

# Continuare da soli, invece di chiedere a qualcuno di ri-sottomettere ogni
# mezz'ora. Il numero della catena viaggia con il job e la limita: se qualcosa
# va storto in modo ripetibile, lo studio si ferma da solo invece di riempire
# la coda.
study_resubmit()
{
    local script="$STUDY_ROOT/scripts/study/$STUDY_PHASE.sh"

    if [[ "${AUTO_RESUBMIT:-$STUDY_CHAINABLE}" != "1" ]]; then
        echo "ri-sottomissione disattivata: rilancia con  qsub $script"
        return 0
    fi
    if [[ "$STUDY_CHAIN" -ge "$STUDY_CHAIN_MAX" ]]; then
        echo "catena arrivata a $STUDY_CHAIN job: mi fermo qui per prudenza."
        echo "se e' normale, rilancia con  STUDY_CHAIN=1 qsub $script"
        return 0
    fi
    if ! command -v qsub > /dev/null || [[ -z "${PBS_JOBID:-}" ]]; then
        echo "fuori da PBS: rilancia con  ./scripts/study/$STUDY_PHASE.sh"
        return 0
    fi

    local next
    mkdir -p "$STUDY_BASE/pbs"
    if next="$(qsub -V -o "$STUDY_BASE/pbs/" \
                    -v "STUDY_CHAIN=$(( STUDY_CHAIN + 1 ))" "$script" 2>&1)"; then
        echo "continua nel job $next"
    else
        echo "ri-sottomissione fallita: $next"
        echo "rilancia a mano con  qsub $script"
    fi
}

# ------------------------------------------------------------------ macchina

study_machine()
{
    STUDY_SOCKETS=$(lscpu | awk -F: '/^Socket\(s\)/ {gsub(/ /,"",$2); print $2}')
    STUDY_PER_SOCKET=$(lscpu | awk -F: '/^Core\(s\) per socket/ {gsub(/ /,"",$2); print $2}')
    : "${STUDY_SOCKETS:=1}" "${STUDY_PER_SOCKET:=1}"

    # Quante CPU ha il nodo, e quante ne ha date a noi. Non sono la stessa
    # cosa e distinguerle e' tutto:
    #
    #   nproc --all      le CPU del nodo, sempre, qualunque cosa ci abbiano dato
    #   NCPUS            quelle che PBS ha assegnato a questo job
    #   nproc            NON e' affidabile qui: rispetta OMP_NUM_THREADS, che
    #                    PBS imposta al numero di CPU del chunk, e su un nodo
    #                    da 112 CPU con 7 assegnate rispondeva 7 -- numero
    #                    giusto per caso e per il motivo sbagliato.
    #   Cpus_allowed     su questo cluster vale 0-111 anche quando le CPU
    #                    concesse sono 7: la maschera di affinita' non e'
    #                    ristretta, quindi da sola non dice se il nodo e' nostro.
    local node_logical="$(nproc --all)"
    STUDY_LOGICAL="${NCPUS:-$node_logical}"
    STUDY_PHYSICAL=$(( STUDY_SOCKETS * STUDY_PER_SOCKET ))
    [[ "$STUDY_LOGICAL" -lt "$STUDY_PHYSICAL" ]] && STUDY_PHYSICAL="$STUDY_LOGICAL"

    echo "=== macchina ==="
    printf 'nodo:          %s\n' "$(hostname -s)"
    printf 'socket:        %s\n' "$STUDY_SOCKETS"
    printf 'cpu del nodo:  %s logiche, %s core fisici\n' \
        "$node_logical" "$(( STUDY_SOCKETS * STUDY_PER_SOCKET ))"
    printf 'cpu concesse:  %s\n' "$STUDY_LOGICAL"
    printf 'nodi NUMA:     %s\n' \
        "$(find /sys/devices/system/node -maxdepth 1 -name 'node[0-9]*' 2>/dev/null | wc -l)"
    printf 'memoria:       %s\n' "$(awk '/MemTotal/ {printf "%.0f GB", $2/1048576}' /proc/meminfo)"
    printf 'mpirun:        %s\n' "$(command -v "${MPIRUN:-mpirun}" || echo assente)"
    printf 'versione MPI:  %s\n' \
        "$(${MPIRUN:-mpirun} --version 2>&1 | head -1)"

    # Il nodo e' davvero tutto nostro? Su questo cluster la risposta e' stata
    # no due volte su tre, e le misure raccolte in quei casi erano inservibili
    # senza che niente nell'output lo dicesse.
    if [[ "$STUDY_LOGICAL" -ge "$node_logical" ]]; then
        STUDY_EXCLUSIVE=1
        printf 'esclusivo:     si\n'
    else
        STUDY_EXCLUSIVE=0
        printf '\n  ATTENZIONE: hai %s CPU su %s. Le misure che seguono sono\n' \
            "$STUDY_LOGICAL" "$node_logical"
        printf '  contaminate dai job dei vicini e non vanno usate per i tempi.\n'
        printf '  Serve:  qsub -q scalability -l select=1:ncpus=%s\n\n' \
            "$node_logical"
    fi
    echo
}

# --------------------------------------------------------------- compilazione
#
# Il Makefile e' l'unica fonte di verita' sui flag: qui si passano solo le
# variabili che esso documenta, e il binario prodotto si sposta nella cache col
# nome della variante. Cosi' una differenza fra studio e build normale non puo'
# nascere per divergenza di flag copiati a mano.

study_build()
{
    local backend="$1" simd="$2" omp="$3" mpi="$4" batch="$5"
    local target="${6:-bench}"
    local key="$target-$backend-simd$simd-omp$omp-mpi$mpi-b$batch"
    local out="$STUDY_BIN/$key"

    if [[ -x "$out" && "${REBUILD:-0}" != "1" ]]; then
        printf '%s' "$out"
        return 0
    fi

    if ! make -s -B -C "$STUDY_ROOT" \
            TRIDIAG="$backend" SIMD="$simd" OMP="$omp" MPI="$mpi" \
            PIPELINE_BATCH_LINES="$batch" \
            "build/tests/$target" > "$STUDY_BIN/$key.build.log" 2>&1; then
        echo "compilazione fallita: $key" >&2
        sed 's/^/    /' "$STUDY_BIN/$key.build.log" | head -20 >&2
        return 1
    fi
    mv "$STUDY_ROOT/build/tests/$target" "$out"
    printf '%s' "$out"
}

# Il file di configurazione di una taglia: griglia e passi, niente altro.
study_config()
{
    local nx="$1" ny="$2" nz="$3" steps="$4"
    local path="$STUDY_OUT/grid-${nx}x${ny}x${nz}-s${steps}.txt"

    printf 'width = %s\nheight = %s\ndepth = %s\nsteps = %s\nt_end = 1e-1\n' \
        "$nx" "$ny" "$nz" "$steps" > "$path"
    printf '%s' "$path"
}

# --------------------------------------------------------------- piazzamento
#
# Meta' del risultato, e la meta' che nessun output segnala quando e' sbagliata.
# Per default mpirun inchioda ogni processo a un core solo e i suoi thread se
# lo spartiscono: la colonna dei thread misura allora zero guadagno, ed e' un
# errore che sembra un risultato (MULTITHREAD.md §8.1).

study_placement()
{
    local ranks="$1" threads="$2"

    : "${STUDY_SOCKETS:=1}" "${STUDY_LOGICAL:=$(nproc)}"
    STUDY_MPI_OPTS=()
    STUDY_OMP_BIND="close"
    STUDY_OMP_WAIT="active"
    STUDY_NOTE=""

    if [[ "$ranks" -eq 1 ]]; then
        # Un rank solo deve coprire tutti i socket, e a distribuirlo e' OpenMP:
        # --bind-to none toglie di mezzo MPI, spread manda i thread su tutti i
        # canali di memoria invece che su quelli di un socket solo.
        STUDY_MPI_OPTS=(--bind-to none)
        STUDY_OMP_BIND="spread"
    elif [[ "$threads" -eq 1 ]]; then
        STUDY_MPI_OPTS=(--map-by core --bind-to core)
    elif [[ $(( ranks % STUDY_SOCKETS )) -eq 0 ]]; then
        # Un gruppo di rank per socket, e i thread di ciascuno dentro il
        # proprio socket: senza questo i thread di rank diversi si mescolano
        # sui due socket e si misura quel disastro invece del dominio diviso.
        STUDY_MPI_OPTS=(--map-by "ppr:$(( ranks / STUDY_SOCKETS )):socket:PE=$threads"
                        --bind-to core)
    else
        STUDY_MPI_OPTS=(--bind-to none)
        STUDY_OMP_BIND="spread"
        STUDY_NOTE="piazzamento non controllato ($ranks rank su $STUDY_SOCKETS socket)"
    fi

    if [[ $(( ranks * threads )) -gt "$STUDY_LOGICAL" ]]; then
        # Piu' unita' che cpu: mpirun rifiuta di legare i processi ai core e
        # fallirebbe prima di partire. Il caso non e' comunque confrontabile
        # con gli altri, e la nota nel CSV lo dice.
        STUDY_MPI_OPTS=(--oversubscribe --bind-to none)
        STUDY_NOTE="${STUDY_NOTE:+$STUDY_NOTE; }in sovrannumero, senza binding"
        # Con piu' thread che core, l'attesa attiva di OpenMP brucia i core
        # contendendoli a chi sta lavorando: un caso in sovrannumero puo'
        # rallentare di ordini di grandezza invece che linearmente. `passive'
        # li fa dormire. Vale solo qui: dove la misura conta, l'attesa attiva
        # e' quella giusta.
        STUDY_OMP_WAIT="passive"
    fi

    # Una fase che sa meglio come vanno piazzati i processi -- il multi-nodo,
    # dove serve un hostfile e una mappatura per nodo -- sostituisce tutto.
    if [[ -n "${PLACEMENT_OVERRIDE:-}" ]]; then
        STUDY_MPI_OPTS=($PLACEMENT_OVERRIDE)
        STUDY_NOTE="${STUDY_NOTE:+$STUDY_NOTE; }piazzamento imposto dalla fase"
    fi
}

# ------------------------------------------------------------------- un caso
#
# study_case backend=schur ranks=8 threads=1 grid="256 256 256" shape="2 2 2"
#
# Argomenti nella forma chiave=valore, tutti facoltativi: quelli non dati
# vengono dai default della fase (CASE_*). Le chiavi sconosciute fermano lo
# script, perche' un parametro scritto male e' un risultato sbagliato.

study_case()
{
    local backend="${CASE_BACKEND:-schur}"
    local batch="${CASE_BATCH:-64}"
    local simd="${CASE_SIMD:-1}"
    local omp="${CASE_OMP:-1}"
    local mpi="${CASE_MPI:-1}"
    local ranks="${CASE_RANKS:-1}"
    local threads="${CASE_THREADS:-1}"
    local grid="${CASE_GRID:-128 128 128}"
    local shape="${CASE_SHAPE:-}"
    local steps="${CASE_STEPS:-10}"
    local repeats="${CASE_REPEATS:-${REPEATS:-2}}"
    local norms="${CASE_NORMS:-0}"
    local label=""
    local note=""
    local wrap=""
    local bind=""
    local arg

    for arg in "$@"; do
        case "$arg" in
            backend=*) backend="${arg#*=}" ;;
            batch=*)   batch="${arg#*=}" ;;
            simd=*)    simd="${arg#*=}" ;;
            omp=*)     omp="${arg#*=}" ;;
            mpi=*)     mpi="${arg#*=}" ;;
            ranks=*)   ranks="${arg#*=}" ;;
            threads=*) threads="${arg#*=}" ;;
            grid=*)    grid="${arg#*=}" ;;
            shape=*)   shape="${arg#*=}" ;;
            steps=*)   steps="${arg#*=}" ;;
            repeats=*) repeats="${arg#*=}" ;;
            norms=*)   norms="${arg#*=}" ;;
            label=*)   label="${arg#*=}" ;;
            note=*)    note="${arg#*=}" ;;
            wrap=*)    wrap="${arg#*=}" ;;
            bind=*)    bind="${arg#*=}" ;;
            *) echo "study_case: argomento sconosciuto: $arg" >&2; return 2 ;;
        esac
    done

    local nx ny nz
    read -r nx ny nz <<< "$grid"
    [[ -z "$label" ]] && label="$backend/r${ranks}t${threads}/${nx}"

    # La chiave della ripresa contiene tutto cio' che cambia la misura: due
    # casi con la stessa chiave sono lo stesso caso.
    local key="$STUDY_PHASE|$label|$backend|$batch|$simd|$omp|$mpi|$ranks|$threads|$nx|$ny|$nz|$steps|${shape// /-}|${wrap// /-}|$bind"

    if [[ "${RESUME:-1}" == "1" ]] && grep -qxF "$key" "$STUDY_KEYS"; then
        STUDY_SKIPPED=$(( STUDY_SKIPPED + 1 ))
        printf '  %-34s gia\x27 fatto\n' "$label"
        return 0
    fi

    # Il tempo che resta decide se questo caso si comincia. Cominciarne uno che
    # non fa in tempo a finire vuol dire farsi uccidere dal walltime nel mezzo:
    # la misura non finisce nel CSV e il tempo e' buttato.
    local remaining=$(( STUDY_BUDGET - ( $(date +%s) - STUDY_STARTED ) ))
    if [[ "${DRY_RUN:-0}" != "1" && "$remaining" -lt "${CASE_MIN_TIME:-90}" ]]; then
        if [[ "$STUDY_OUT_OF_TIME" -eq 0 ]]; then
            printf '  budget finito: i casi che restano vanno al prossimo job\n'
            STUDY_OUT_OF_TIME=1
        fi
        STUDY_PENDING=$(( STUDY_PENDING + 1 ))
        return 0
    fi

    printf '  %-34s ' "$label"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf 'backend=%s batch=%s simd=%s omp=%s ranks=%s thr=%s grid=%sx%sx%s shape=%s steps=%s\n' \
            "$backend" "$batch" "$simd" "$omp" "$ranks" "$threads" \
            "$nx" "$ny" "$nz" "${shape:-auto}" "$steps"
        STUDY_CASES=$(( STUDY_CASES + 1 ))
        return 0
    fi

    local exe config
    if ! exe="$(study_build "$backend" "$simd" "$omp" "$mpi" "$batch")"; then
        study_record "$label" "$backend" "$batch" "$simd" "$omp" "$mpi" \
            "$ranks" "$threads" "$nx" "$ny" "$nz" "$steps" "" \
            "build" "compilazione fallita"
        printf 'compilazione fallita\n'
        STUDY_FAILED=$(( STUDY_FAILED + 1 ))
        return 0
    fi
    config="$(study_config "$nx" "$ny" "$nz" "$steps")"

    study_placement "$ranks" "$threads"
    # bind= forza il piazzamento dei thread quando lo studio e' proprio quello:
    # gli stessi thread su un socket o su due sono la stessa potenza di calcolo
    # e meta' o tutti i canali di memoria.
    [[ -n "$bind" ]] && STUDY_OMP_BIND="$bind"
    [[ -n "$STUDY_NOTE" ]] && note="${note:+$note; }$STUDY_NOTE"

    local best_wall="" best_line="" status="ok" started
    local repeat out line wall
    started="$(date +%s)"

    # Nessun caso puo' durare piu' del budget che resta: oltre quello lo
    # ucciderebbe comunque il walltime, e senza lasciare traccia.
    local case_timeout="${CASE_TIMEOUT:-1500}"
    [[ "$remaining" -lt "$case_timeout" ]] && case_timeout="$remaining"

    # Un binario senza MPI non si lancia con mpirun: si esegue e basta,
    # inchiodato a un core solo perche' il caso seriale e' un riferimento e
    # non deve migrare fra i socket a meta' misura.
    local -a command
    if [[ "$mpi" == "0" ]]; then
        command=()
        if [[ "$threads" -eq 1 ]] && command -v taskset > /dev/null; then
            command=(taskset -c 0)
        fi
        [[ -n "$wrap" ]] && command+=($wrap)
        command+=("$exe" "$config")
    else
        command=("${MPIRUN:-mpirun}" "${STUDY_MPI_OPTS[@]}"
                 -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND
                 -x OMP_WAIT_POLICY -x BENCH_NORMS
                 -n "$ranks")
        # Il wrapper sta fra mpirun e l'eseguibile: e' ogni rank a doverne
        # ereditare i vincoli, non il lanciatore.
        [[ -n "$wrap" ]] && command+=($wrap)
        command+=("$exe" "$config")
        [[ -n "$shape" ]] && command+=($shape)
    fi

    for (( repeat = 0; repeat < repeats; repeat++ )); do
        set +e
        out="$(OMP_NUM_THREADS="$threads" \
               OMP_PLACES="${OMP_PLACES:-cores}" \
               OMP_PROC_BIND="$STUDY_OMP_BIND" \
               OMP_WAIT_POLICY="$STUDY_OMP_WAIT" \
               BENCH_NORMS="$norms" \
               timeout --kill-after=30 "$case_timeout" \
               "${command[@]}" 2>&1)"
        local code=$?
        set -e

        if [[ $code -eq 124 || $code -eq 137 ]]; then
            status="timeout"
            break
        fi
        if [[ $code -ne 0 ]]; then
            status="fallito"
            printf '\n    uscita %d, ecco cosa ha stampato:\n' "$code"
            sed 's/^/      /' <<< "$out" | tail -15
            break
        fi

        line="$(study_parse <<< "$out")" || line=""
        if [[ -z "$line" ]]; then
            status="illeggibile"
            printf '\n    nessun tempo nell\x27output:\n'
            sed 's/^/      /' <<< "$out" | tail -15
            break
        fi

        # px,py,pz sono i primi tre campi: il tempo e' il quarto.
        wall="$(cut -d, -f4 <<< "$line")"
        if [[ -z "$best_wall" ]] || awk "BEGIN{exit !($wall < $best_wall)}"; then
            best_wall="$wall"
            best_line="$line"
        fi
    done

    local seconds=$(( $(date +%s) - started ))

    if [[ "$status" != "ok" ]]; then
        STUDY_FAILED=$(( STUDY_FAILED + 1 ))
        study_record "$label" "$backend" "$batch" "$simd" "$omp" "$mpi" \
            "$ranks" "$threads" "$nx" "$ny" "$nz" "$steps" "" \
            "$status" "$note"
        printf '%s (%ds)\n' "$status" "$seconds"
        return 0
    fi

    study_record "$label" "$backend" "$batch" "$simd" "$omp" "$mpi" \
        "$ranks" "$threads" "$nx" "$ny" "$nz" "$steps" "$best_line" \
        ok "$note"
    printf '%s' "$key" >> "$STUDY_KEYS"
    printf '\n' >> "$STUDY_KEYS"
    STUDY_CASES=$(( STUDY_CASES + 1 ))

    local shape_out untimed rss
    shape_out="$(cut -d, -f1-3 <<< "$best_line" | tr ',' 'x')"
    untimed="$(cut -d, -f14 <<< "$best_line")"
    rss="$(cut -d, -f16 <<< "$best_line")"
    printf '%10s ms  %-9s non contato %6s ms  rss %7s MB  (%ds)\n' \
        "$best_wall" "$shape_out" "$untimed" "$rss" "$seconds"
}

# Legge l'output di bench. I pattern sono ancorati: "eta system" senza ancora
# matcherebbe anche "zeta system", e si leggerebbero i tempi di zeta credendoli
# di eta -- numeri plausibili, di un'altra cosa (MULTITHREAD.md §8.3).
study_parse()
{
    awk '
        /^  eta system/     { eta  = $3 }
        /^  zeta system/    { zeta = $3 }
        /^  u system/       { u    = $3 }
        /^  psi system/     { psi  = $3 }
        /^  phi low/        { lo   = $3 }
        /^  phi high/       { hi   = $3 }
        /^  pressure:/      { pr   = $3 }
        /^  porosity:/      { po   = $3 }
        /^  wall per step/  { wall = $4 }
        /^  mpi per step/   { mpi  = $4 }
        /^  per cell-step/  { cell = $3 }
        /^  bench proc grid/   { px = $4; py = $6; pz = $8 }
        /^  bench peak rss/    { rss = $4 }
        /^  L2 error u_x/   { lux = $4 }
        /^  L2 error p/     { lp  = $4 }
        END {
            if (wall == "") { exit 1 }
            sum = eta + zeta + u + psi + lo + hi + pr + po
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.3f,%s,%s,%s,%s",
                   px, py, pz, wall, mpi, eta, zeta, u, psi, lo, hi, pr, po,
                   wall - sum, cell, rss, lux, lp
        }'
}

# Una riga di CSV, sempre con lo stesso numero di colonne anche quando il caso
# non ha prodotto niente: un buco che sfasa le colonne si scopre settimane dopo.
study_record()
{
    local label="$1" backend="$2" batch="$3" simd="$4" omp="$5" mpi="$6"
    local ranks="$7" threads="$8" nx="$9" ny="${10}" nz="${11}" steps="${12}"
    local measured="${13}" status="${14}" note="${15}"

    if [[ -z "$measured" ]]; then
        measured="$(printf ',%.0s' $(seq 2 "$STUDY_MEASURED_FIELDS"))"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$STUDY_PHASE" "$label" "$backend" "$batch" "$simd" "$omp" "$mpi" \
        "$ranks" "$threads" "$nx" "$ny" "$nz" "$steps" "$measured" \
        "$status" "${note//,/;}" >> "$STUDY_CSV"
}

# Le due forme di griglia di processi che questo studio usa piu' spesso.
# MPI_Dims_create le sceglie cosi', e averle scritte qui permette di imporre
# la stessa forma anche dove la si vuole diversa dal default.
study_auto_shape()
{
    case "$1" in
        1)  echo "1 1 1" ;;
        2)  echo "2 1 1" ;;
        4)  echo "2 2 1" ;;
        7)  echo "7 1 1" ;;
        8)  echo "2 2 2" ;;
        14) echo "7 2 1" ;;
        16) echo "4 2 2" ;;
        28) echo "7 2 2" ;;
        56) echo "7 4 2" ;;
        112) echo "7 4 4" ;;
        *)  echo "" ;;
    esac
}
