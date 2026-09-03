#!/usr/bin/env bash
#PBS -N nsb-06-batch
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Fase 6 -- PIPELINE_BATCH_LINES, l'unica manopola della pipeline, mai misurata.
#
# Domanda: quante linee conviene far viaggiare insieme?
#
# La tensione e' scritta in PIPELINE.md §1.6 e non e' stata risolta da nessuna
# misura:
#
#   T(B) ~= S*W  +  (P-1)*W/B  +  S*B*lambda
#           lavoro   riempimento    latenza
#                    (cala con B)   (cresce con B)
#
# Batch piccoli riempiono la catena subito ma spediscono tanti messaggi; batch
# grandi il contrario. Il default 64 e' una scelta empirica, dichiarata tale:
# "non il risultato di questa formula". §4.2 sospetta che valori piu' grandi
# convengano e chiude con "va misurato".
#
# C'e' un terzo costo che la formula non contiene e che qui si misura insieme
# agli altri: la memoria. Lo scratch e' arrotondato a batch interi, quindi
# batch grandi su pochi lati sprecano; e comunque la pipeline tiene c' e d' di
# tutto il blocco locale, che schur non paga affatto (PIPELINE.md §1.7). La
# colonna rss e' parte del risultato, non un contorno.
#
# La forma entra nello studio perche' e' lei a fissare P e il numero di linee,
# cioe' entrambi i termini della formula: lo stesso batch su 8x1x1 e su 1x1x8
# non e' lo stesso batch.
#
#   qsub scripts/study/06_batch.sh

# PBS non esegue questo file dove sta: ne mette una COPIA nella propria
# directory di spool e lancia quella. $BASH_SOURCE punta li', quindi non dice
# niente su dove sia il repo -- a dirlo e' PBS_O_WORKDIR, la directory da cui
# si e' fatto qsub. Cercare lib.sh accanto allo script funziona da riga di
# comando e fallisce dentro un job, prima ancora che esista un log in cui
# vederlo.
cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1

if [[ ! -f scripts/study/lib.sh ]]; then
    echo "qsub va fatto dalla radice del repo; qui sono in $PWD" >&2
    exit 1
fi

source scripts/study/lib.sh
study_begin 06_batch
study_machine

GRID="${GRID:-224 224 224}"
STEPS="${STEPS:-10}"
BATCHES="${BATCHES:-8 16 32 64 128 256 512 1024}"
RANKS="${RANKS:-2 8 56}"

CASE_BACKEND=pipeline
CASE_THREADS=1
CASE_OMP=0
CASE_MPI=1
CASE_SIMD=1
CASE_GRID="$GRID"
CASE_STEPS="$STEPS"
CASE_REPEATS="${REPEATS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-900}"

for ranks in $RANKS; do
    shape="$(study_auto_shape "$ranks")"
    echo "=== $ranks processi, forma ${shape// /x}, ${GRID// /x} ==="
    for batch in $BATCHES; do
        study_case label="R=$ranks batch=$batch" ranks="$ranks" \
            shape="$shape" batch="$batch"
    done
    echo
done

# Lo stesso batch su due forme opposte: 8x1x1 divide l'asse X (linee lunghe,
# tante), 1x1x8 divide Z. Cambiano sia P sia il numero di linee, cioe' i due
# termini che la formula mette uno contro l'altro.
cross="${CROSS_RANKS:-8}"
echo "=== $cross processi: lo stesso batch su due assi diversi ==="
for shape in "$cross 1 1" "1 1 $cross"; do
    for batch in ${CROSS_BATCHES:-16 64 256 1024}; do
        study_case label="${cross}p ${shape// /x} batch=$batch" ranks="$cross" \
            shape="$shape" batch="$batch"
    done
done
echo

# Un riferimento schur nelle stesse condizioni: la manopola migliore della
# pipeline va confrontata con l'algoritmo che non ne ha nessuna.
echo "=== riferimento: schur, stesse condizioni, nessuna manopola ==="
for ranks in $RANKS; do
    study_case label="schur R=$ranks" backend=schur ranks="$ranks" \
        shape="$(study_auto_shape "$ranks")" batch=64
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== risultato ==="
    awk -F, -v phase=06_batch '
    NR == 1 || $1 != phase || $32 != "ok" { next }
    {
        form = $14 "-" $15 "-" $16
        k = $3 "," $8 "," form "," $4
        wall[k] = $17; mpims[k] = $18; rss[k] = $29
        nx[k] = $10; ny[k] = $11; nz[k] = $12
        px[k] = $14; py[k] = $15; pz[k] = $16
        gk = $8 "," form
        if (!(gk in seen_g)) { gs[++ng] = gk; seen_g[gk] = 1 }
        if ($3 == "pipeline" && !(gk "," $4 in seen_b)) {
            bl[gk] = bl[gk] " " $4; seen_b[gk "," $4] = 1
        }
    }
    # Efficienza teorica del riempimento, PIPELINE.md §1.5:
    #   B = linee locali / batch,  P = processi lungo l\x27asse diviso
    #   momento (tre componenti incatenate) = 3B / (3B + P - 1)
    function fill_efficiency(k, batch,   a, best, lines, B, P, e) {
        best = 1.0
        for (a = 0; a < 3; a++) {
            P = (a == 0 ? px[k] : (a == 1 ? py[k] : pz[k]))
            if (P < 2) continue
            if (a == 0)      lines = (ny[k] / py[k]) * (nz[k] / pz[k])
            else if (a == 1) lines = (nx[k] / px[k]) * (nz[k] / pz[k])
            else             lines = (nx[k] / px[k]) * (ny[k] / py[k])
            B = int((lines + batch - 1) / batch)
            e = 3 * B / (3 * B + P - 1)
            if (e < best) best = e
        }
        return best
    }
    END {
        for (i = 1; i <= ng; i++) {
            gk = gs[i]; split(gk, p, ",")
            printf "\n  %s processi, forma %s\n", p[1], p[2]
            printf "  %-8s %10s %8s %7s %10s %12s\n",
                   "batch", "ms/passo", "vs 64", "%mpi", "rss MB", "eff. teorica"
            n = split(bl[gk], list, " ")
            for (a = 1; a <= n; a++) for (b = a + 1; b <= n; b++)
                if (list[a] + 0 > list[b] + 0) { t = list[a]; list[a] = list[b]; list[b] = t }
            ref = wall["pipeline," gk ",64"]
            best = 0
            for (j = 1; j <= n; j++) {
                batch = list[j]; k = "pipeline," gk "," batch
                if (!(k in wall)) continue
                printf "  %-8s %10.1f %7.2fx %6.0f%% %10.0f %11.1f%%\n",
                       batch, wall[k], (ref ? ref / wall[k] : 0),
                       100 * mpims[k] / wall[k], rss[k],
                       100 * fill_efficiency(k, batch)
                if (best == 0 || wall[k] < best) { best = wall[k]; bb = batch }
            }
            ks = "schur," gk ",64"
            if (ks in wall)
                printf "  %-8s %10.1f %7.2fx %6.0f%% %10.0f %11s\n",
                       "schur", wall[ks], (ref ? ref / wall[ks] : 0),
                       100 * mpims[ks] / wall[ks], rss[ks], "n/d"
            if (best > 0) printf "    batch migliore: %s\n", bb
        }
        print ""
        print "  `eff. teorica\x27 e\x27 3B/(3B+P-1) sull\x27asse diviso peggiore:"
        print "  quanto la formula di PIPELINE.md §1.5 si aspetta di perdere nel"
        print "  transitorio. Se resta sopra il 99% mentre il tempo cambia molto,"
        print "  a decidere non e\x27 il riempimento ma la latenza o la memoria --"
        print "  ed e\x27 esattamente la domanda che §4.2 lasciava aperta."
    }' "$STUDY_CSV"
fi

study_end
