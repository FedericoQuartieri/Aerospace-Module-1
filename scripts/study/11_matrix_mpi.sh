#!/usr/bin/env bash
#PBS -N nsb-11-mpi
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Fase 11 -- un thread per rank, e OGNI forma della griglia di processi.
#
# La 03 misura i rank crescenti con una forma sola per numero, scelta a mano;
# la 04 confronta qualche forma. Questa le prende tutte: per ogni numero di
# rank, tutte le terne (px, py, pz) con prodotto quel numero. Sono 80 forme in
# tutto fra 1 e 56 rank.
#
# Le permutazioni non sono doppioni, ed e' il punto della fase. I tre assi non
# sono intercambiabili:
#
#   x  e' la direzione contigua in memoria. Schur, quando la divide, perde i
#      kernel a linea intera lungo le altre due? no: perde quelli lungo x, che
#      non ha, e paga tre risoluzioni locali invece di una. La pipeline ci
#      mette le linee NON adiacenti, quindi lungo x non vettorizza mai.
#   y, z  sono le direzioni dove i kernel vettoriali lavorano, e dividerle
#      significa perderli -- per Schur. Per la pipeline no: il suo kernel
#      lavora attraverso le linee e sopravvive alla divisione.
#
# Quindi la stessa quantita' di processi, disposta in modo diverso, dovrebbe
# dare tempi diversi, e diversi in modo OPPOSTO per i due backend. Se non
# succede, l'effetto e' sotto al rumore della banda di memoria e va detto.
#
# ----------------------------------------------------------------------------
# Perche' la griglia cubica da sola non basta
# ----------------------------------------------------------------------------
#
# Su una griglia globale cubica, la forma (1, 1, 56) non da' soltanto "z
# diviso": da' anche un blocco locale di 224 x 224 x 4, cioe' una lamina. Il
# tempo che ne esce mescola due cose -- quale asse e' tagliato e che forma ha
# preso il blocco -- e dal numero non si possono separare.
#
# Per questo la fase misura la stessa domanda in tre modi, e sono tre blocchi
# diversi qui sotto:
#
#   1. griglia globale cubica    com'e' in produzione. E' il caso reale, e i
#                                due effetti restano insieme.
#   2. blocco locale cubico      la griglia globale segue la forma: con B = 96
#                                e forma (p, q, r) il globale e' 96p x 96q x
#                                96r. Ogni forma ha allora lo STESSO blocco
#                                locale e lo STESSO lavoro per processo, e
#                                l'unica cosa che cambia e' quale asse porta i
#                                tagli. E' qui che l'effetto dell'asse si legge
#                                da solo.
#   3. forma del blocco          forma dei processi fissa, griglia globale
#                                stirata: stesso numero di celle per processo,
#                                blocco locale lungo o piatto. E' l'altra meta'
#                                di cio' che il caso 1 teneva insieme.
#
#   qsub scripts/study/11_matrix_mpi.sh
#   RANKS="8 56" qsub scripts/study/11_matrix_mpi.sh

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub va fatto dalla radice del repo; qui sono in $PWD" >&2
    exit 1
fi

source scripts/study/lib.sh
source scripts/study/matrix.sh

STUDY_CHAIN_MAX="$MATRIX_CHAIN_MAX"
study_begin 11_matrix_mpi
study_machine

# La taglia dove si prova tutto, forme per entrambi i SIMD.
FULL_GRID="${FULL_GRID:-224}"
# Le taglie dove si provano tutte le forme ma solo con SIMD acceso: servono a
# vedere se la classifica delle forme dipende dalla taglia.
PLAIN_GRIDS="${PLAIN_GRIDS:-128}"
RANKS="${RANKS:-$MATRIX_RANKS}"
# Blocco 2: il lato del blocco locale, che resta lo stesso per ogni forma.
BLOCK="${BLOCK:-96}"
# Blocco 3: blocchi locali di uguale volume e forma diversa, a forma dei
# processi fissa. 2 097 152 celle per processo in tutte le righe.
ASPETTI="${ASPETTI:-128 128 128|256 128 64|64 128 256|512 64 64|64 512 64|64 64 512}"
ASPETTO_RANKS="${ASPETTO_RANKS:-8}"

CASE_THREADS=1
# Senza OpenMP: qui i thread non c'entrano, e la build a thread costa circa
# l'1% anche a un thread solo. Le righe ponte in fondo ricuciono con la 12.
CASE_OMP=0
CASE_MPI=1
CASE_REPEATS="${REPEATS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-900}"

emit_shapes()
{
    local n="$1" grid="$2" simd="$3" steps="$4" backend shape

    for backend in $MATRIX_BACKENDS; do
        while read -r shape; do
            matrix_shape_fits "$shape" "$grid" || continue
            study_case label="$backend R=$n ${shape// /x} s$simd" \
                backend="$backend" ranks="$n" shape="$shape" simd="$simd" \
                grid="$grid" steps="$steps"
        done < <(matrix_shapes "$n")
    done
}

steps="$(matrix_steps "$FULL_GRID")"
grid="$FULL_GRID $FULL_GRID $FULL_GRID"
for simd in $MATRIX_SIMD; do
    echo "=== ${FULL_GRID}^3, simd=$simd: ogni forma di ogni numero di rank ==="
    for n in $RANKS; do
        emit_shapes "$n" "$grid" "$simd" "$steps"
    done
    echo
done

for m in $PLAIN_GRIDS; do
    steps="$(matrix_steps "$m")"
    grid="$m $m $m"
    echo "=== ${m}^3, simd=1: le stesse forme su una taglia diversa ==="
    for n in $RANKS; do
        emit_shapes "$n" "$grid" 1 "$steps"
    done
    echo
done

# --------------------------------------------------------------- blocco 2
#
# Il globale segue la forma, quindi ogni riga ha lo stesso blocco locale e lo
# stesso lavoro per processo. Se l'asse tagliato non contasse, queste righe
# darebbero tutte lo stesso tempo: e' un confronto a lavoro costante, non a
# problema costante, ed e' l'unico modo di leggere l'asse da solo.

echo "=== blocco locale ${BLOCK}^3 fisso: cambia solo quale asse e' tagliato ==="
steps="$(matrix_steps "$BLOCK")"
for backend in $MATRIX_BACKENDS; do
    for n in $RANKS; do
        while read -r px py pz; do
            study_case label="cubo $backend R=$n ${px}x${py}x${pz}" \
                backend="$backend" ranks="$n" shape="$px $py $pz" simd=1 \
                grid="$(( px * BLOCK )) $(( py * BLOCK )) $(( pz * BLOCK ))" \
                steps="$steps" \
                note="blocco locale ${BLOCK}^3, globale al seguito"
        done < <(matrix_shapes "$n")
    done
done
echo

# --------------------------------------------------------------- blocco 3
#
# L'altra meta': forma dei processi fissa, blocco locale di uguale volume e
# proporzioni diverse. Qui l'asse tagliato non cambia mai, cambia solo se il
# blocco e' un cubo, una barra o una lamina -- e su uno stencil quella forma
# decide quante linee di cache si riusano.

echo "=== stesso volume per processo, blocco locale di forma diversa ==="
IFS='|' read -r -a aspetti <<< "$ASPETTI"
for n in $ASPETTO_RANKS; do
    shape="$(study_auto_shape "$n")"
    [[ -n "$shape" ]] || continue
    read -r px py pz <<< "$shape"
    for entry in "${aspetti[@]}"; do
        read -r bx by bz <<< "$entry"
        for backend in $MATRIX_BACKENDS; do
            for simd in $MATRIX_SIMD; do
                study_case label="aspetto $backend ${bx}x${by}x${bz} s$simd" \
                    backend="$backend" ranks="$n" shape="$shape" simd="$simd" \
                    grid="$(( px * bx )) $(( py * by )) $(( pz * bz ))" \
                    steps="$(matrix_steps "$bx")" \
                    note="blocco locale ${bx}x${by}x${bz}"
            done
        done
    done
done
echo

# Righe ponte verso la 12: stessa cosa, build con OpenMP e un thread solo.
echo "=== righe ponte: build OpenMP a un thread ==="
for backend in $MATRIX_BACKENDS; do
    for n in 1 8 56; do
        shape="$(study_auto_shape "$n")"
        [[ -n "$shape" ]] || continue
        study_case label="$backend ponte R=$n omp=1" backend="$backend" \
            ranks="$n" shape="$shape" simd=1 omp=1 \
            grid="$FULL_GRID $FULL_GRID $FULL_GRID" \
            steps="$(matrix_steps "$FULL_GRID")"
    done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== la forma migliore e la peggiore, per ogni numero di rank ==="
    awk -F, -v phase=11_matrix_mpi '
    NR == 1 || $1 != phase || $33 != "ok" { next }
    $2 ~ /ponte|^cubo |^aspetto / { next }
    {
        k = $3 "," $5 "," $10 "," $8
        s = $14 "x" $15 "x" $16
        if (!(k in best) || $17 + 0 < best[k]) { best[k] = $17 + 0; bs[k] = s }
        if (!(k in worst) || $17 + 0 > worst[k]) { worst[k] = $17 + 0; ws[k] = s }
        if (!(k in seen)) { keys[++nk] = k; seen[k] = 1 }
    }
    END {
        printf "  %-10s %-5s %-6s %-5s %9s %-9s %9s %-9s %7s\n",
               "backend", "simd", "griglia", "rank",
               "migliore", "forma", "peggiore", "forma", "scarto"
        for (i = 1; i <= nk; i++) {
            split(keys[i], p, ",")
            printf "  %-10s %-5s %-6s %-5s %9.1f %-9s %9.1f %-9s %6.2fx\n",
                   p[1], p[2], p[3], p[4],
                   best[keys[i]], bs[keys[i]],
                   worst[keys[i]], ws[keys[i]],
                   (best[keys[i]] > 0 ? worst[keys[i]] / best[keys[i]] : 0)
        }
        print ""
        print "  Lo scarto e\x27 quanto costa disporre male gli stessi processi."
        print "  Se per i due backend la forma migliore e\x27 la stessa, allora a"
        print "  decidere e\x27 la banda di memoria e non l\x27algoritmo."
        print ""
        print "  ATTENZIONE: qui la griglia globale e\x27 cubica, quindi una forma"
        print "  molto sbilanciata da\x27 anche un blocco locale a lamina. Le due"
        print "  cause si separano nella tabella qui sotto."
    }' "$STUDY_CSV"

    echo
    echo "=== blocco locale fisso: il tempo dei tre tagli puri, per asse ==="
    awk -F, -v phase=11_matrix_mpi '
    NR == 1 || $1 != phase || $33 != "ok" || $2 !~ /^cubo / { next }
    {
        # Taglio puro: tutti i processi su un asse solo.
        asse = ""
        if ($15 == 1 && $16 == 1) asse = "x"
        else if ($14 == 1 && $16 == 1) asse = "y"
        else if ($14 == 1 && $15 == 1) asse = "z"
        if (asse == "" || $8 + 0 < 2) next
        wall[$3 "," $8 "," asse] = $17 + 0
        if (!($3 "," $8 in seen)) { keys[++nk] = $3 "," $8; seen[$3 "," $8] = 1 }
    }
    END {
        printf "  %-10s %-6s %9s %9s %9s %9s\n",
               "backend", "rank", "x", "y", "z", "z/x"
        for (i = 1; i <= nk; i++) {
            split(keys[i], p, ",")
            x = wall[keys[i] ",x"]; y = wall[keys[i] ",y"]; z = wall[keys[i] ",z"]
            printf "  %-10s %-6s", p[1], p[2]
            printf " %9s", (x ? sprintf("%.1f", x) : "-")
            printf " %9s", (y ? sprintf("%.1f", y) : "-")
            printf " %9s", (z ? sprintf("%.1f", z) : "-")
            printf " %9s\n", (x && z ? sprintf("%.2fx", z / x) : "-")
        }
        print ""
        print "  Stesso blocco locale in tutte le colonne, stesso lavoro per"
        print "  processo: se l\x27asse non contasse, le tre colonne sarebbero"
        print "  uguali. Schur e pipeline dovrebbero sbilanciarsi in versi opposti."
    }' "$STUDY_CSV"
fi

study_end
