# shellcheck shell=bash
#
# La parte comune alle fasi 10-15, la campagna esaustiva.
#
# Lo studio che precedeva questa campagna faceva una domanda per fase e
# variava una cosa per volta; e' stato tolto, e queste sei lo contengono.
# Riempiono la matrice: ogni backend per ogni SIMD per ogni numero di rank per
# ogni forma della griglia di processi per ogni numero di thread, e le taglie
# sopra a tutto. Non c'e' una previsione da falsificare, c'e' una superficie da
# misurare, e il suo valore e' che dopo si possono fare domande che oggi non
# sappiamo ancora di avere.
#
# Sei fasi e non una perche' ognuna e' un job PBS a se': chiedono tutte un nodo
# intero, quindi PBS le mette su sei nodi diversi e girano insieme. Ognuna si
# ri-sottomette da sola finche' non ha finito, e ognuna riprende da dove era
# arrivata, cosi' la campagna dura quanto le serve senza che nessuno la segua.
#
#   10_matrix_threads   un rank, thread crescenti: la macchina da sola
#   11_matrix_mpi       un thread, rank crescenti, OGNI forma della griglia
#   12_matrix_hybrid    il prodotto pieno rank x thread
#   13_matrix_batch     il batch della pipeline contro il piazzamento
#   14_matrix_size      la taglia del problema, forte e debole
#   15_matrix_check     le norme: la matrice risolve ancora il problema giusto
#
# Tutte scrivono le stesse colonne di lib.sh, quindi
# `./scripts/run_study.sh merge' le unisce in un CSV solo. `./scripts/plot_matrix.py' disegna da quello.
#
# Quanto dura: con i default sono circa 1800 casi. Distribuiti su sei catene
# in parallelo sono qualche ora ciascuna. Per usare davvero quattro giorni si
# allargano gli assi dall'ambiente, che e' il motivo per cui sono tutti
# variabili:
#
#   GRIDS="128 224 256" REPEATS=3 ./scripts/run_study.sh submit
#
# Un avvertimento che vale per tutta la campagna: il multi-nodo qui non c'e'.
# Su questo cluster non esiste un modo funzionante di lanciare processi su piu'
# nodi -- ne' plm tm, ne' ssh, ne' pbs_tmrsh -- quindi il tetto e' un nodo: 56 core fisici, 112
# cpu logiche. Ogni caso che chiede piu' di 112 unita' non viene emesso.

# I valori che tornano in piu' fasi. Sono divisori di 56 perche' e' il numero
# di core fisici del nodo: numeri che non lo dividono lasciano un socket
# spaiato e misurano il piazzamento invece dell'algoritmo.
MATRIX_RANKS="${MATRIX_RANKS:-1 2 4 7 8 14 28 56}"
MATRIX_THREADS="${MATRIX_THREADS:-1 2 4 7 8 14 28 56 112}"
MATRIX_BACKENDS="${MATRIX_BACKENDS:-schur pipeline}"
MATRIX_SIMD="${MATRIX_SIMD:-0 1}"

# Il tetto di unita' (rank x thread). Di norma le cpu logiche del nodo; si
# abbassa a mano per provare la campagna su una macchina piccola.
matrix_units()
{
    echo "${MATRIX_UNITS:-${STUDY_LOGICAL:-$(nproc --all)}}"
}

# Quanti passi temporali per una taglia. Il prodotto passi x celle e' quello
# che decide quanto dura un caso, e i casi devono durare tutti piu' o meno
# uguale: altrimenti la parte grossa della campagna se ne va in due righe.
matrix_steps()
{
    local n="$1"

    if [[ -n "${STEPS:-}" ]]; then
        echo "$STEPS"
        return
    fi
    if   [[ "$n" -le  64 ]]; then echo 20
    elif [[ "$n" -le 128 ]]; then echo 10
    elif [[ "$n" -le 192 ]]; then echo 6
    else                          echo 4
    fi
}

# TUTTE le forme (px, py, pz) con px*py*pz = n, nell'ordine degli assi.
#
# Le permutazioni NON sono doppioni. I tre assi non sono intercambiabili: x e'
# la direzione contigua in memoria, e i due backend la trattano diversamente --
# Schur ci perde i kernel a linea intera appena la divide, la pipeline ci mette
# le linee non adiacenti. "Quale asse viene diviso" e' esattamente una delle
# variabili che questa campagna deve misurare, non una da quozientare via.
matrix_shapes()
{
    local n="$1" px py pz rest

    for (( px = 1; px <= n; px++ )); do
        (( n % px == 0 )) || continue
        rest=$(( n / px ))
        for (( py = 1; py <= rest; py++ )); do
            (( rest % py == 0 )) || continue
            pz=$(( rest / py ))
            printf '%d %d %d\n' "$px" "$py" "$pz"
        done
    done
}

# Le coppie (rank, thread) che ci stanno nel nodo.
matrix_pairs()
{
    local units r t
    units="$(matrix_units)"

    for r in $MATRIX_RANKS; do
        for t in $MATRIX_THREADS; do
            (( r * t <= units )) && printf '%d %d\n' "$r" "$t"
        done
    done
}

# Una forma non deve chiedere piu' blocchi che celle lungo un asse: decomp
# chiude il programma se lo fa, e sarebbe un caso fallito invece di un caso
# saltato.
matrix_shape_fits()
{
    local px py pz nx ny nz
    read -r px py pz <<< "$1"
    read -r nx ny nz <<< "$2"

    [[ "$px" -le "$nx" && "$py" -le "$ny" && "$pz" -le "$nz" ]]
}

# Le fasi della campagna durano piu' di una catena normale: quattro giorni di
# job da mezz'ora sono quasi duecento anelli. Il tetto resta, perche' serve a
# fermare una catena che gira a vuoto, ma sta molto piu' in alto.
MATRIX_CHAIN_MAX="${MATRIX_CHAIN_MAX:-200}"
