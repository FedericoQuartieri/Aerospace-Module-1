#!/usr/bin/env bash
#PBS -N nsb-05-hybrid
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=08:00:00
#PBS -j oe
#
# Fase 5 -- gli stessi core, spesi in processi o in thread.
#
# Domanda: a parita' di core occupati, dove conviene metterli? E la risposta e'
# la stessa per i due backend?
#
# Su schur la risposta e' gia' nota e sgradevole (MULTITHREAD.md §9.2): a
# 256^3 con rank x thread = 56, le due estremita' costano 274 e 287 ms e tutto
# il centro fra 800 e 3900. Il colpevole e' in momentum.c: quando un asse e'
# diviso, il ciclo sui piani non viene threadato, e per ogni piano si apre un
# team di thread, si assembla, si fa una collettiva, si apre un altro team.
# Con ~256 piani e team da 28 il costo di apertura diventa il termine
# dominante.
#
# La previsione per la pipeline e' che quel crollo NON ci sia, per un motivo
# che non e' un merito: src/tridiag/pipeline/ non contiene nessuna direttiva
# OpenMP, quindi non apre nessun team, quindi non paga nessuna apertura. Sara'
# lenta in modo uniforme invece che a macchie. Se la misura lo conferma, il
# confronto fra le due curve isola il costo delle aperture di team dal costo
# di dividere il dominio: sono due cose che a 128^3 e 4 thread stavano dentro
# lo stesso 17% e nessuno poteva separare.
#
# Tre prodotti invece di uno:
#
#   28    meta' macchina: un socket pieno, niente traffico fra i due
#   56    tutti i core fisici -- la riga di §9.2, da riprodurre
#   112   tutte le cpu logiche: l'SMT non aggiunge unita' aritmetiche, e sul
#         portatile peggiorava. Su un nodo vero va verificato, non dedotto.
#
#   qsub scripts/study/05_hybrid.sh
#   PRODUCTS=56 GRIDS=256 qsub scripts/study/05_hybrid.sh

source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"

cd "${PBS_O_WORKDIR:-$STUDY_ROOT}"
study_begin 05_hybrid
study_machine

GRIDS="${GRIDS:-128 256}"
PRODUCTS="${PRODUCTS:-28 56 112}"

CASE_SIMD=1
CASE_OMP=1
CASE_MPI=1
CASE_TIMEOUT="${CASE_TIMEOUT:-3600}"

# rank x thread per ogni prodotto. Sono tutte le fattorizzazioni sensate: le
# estremita' e tutto quello che c'e' in mezzo, perche' e' proprio il centro
# della tabella la parte interessante.
config_list()
{
    local product="$1" r
    for (( r = 1; r <= product; r++ )); do
        if [[ $(( product % r )) -eq 0 ]]; then
            printf '%dx%d ' "$r" "$(( product / r ))"
        fi
    done
    printf '\n'
}

for n in $GRIDS; do
    steps="${STEPS:-10}"
    [[ "$n" -le 128 ]] && steps="${STEPS_SMALL:-20}"
    # A 256^3 il centro della tabella costa fino a 40 secondi per passo: una
    # sola ripetizione basta a distinguere un fattore 14, e due costerebbero
    # ore per rifinire una cifra che non cambia nessuna conclusione.
    repeats=2
    [[ "$n" -ge 224 ]] && repeats=1

    for product in $PRODUCTS; do
        echo "=== ${n}^3, rank x thread = $product, $steps passi ==="
        for backend in schur pipeline; do
            for spec in $(config_list "$product"); do
                ranks="${spec%x*}"
                threads="${spec#*x}"
                shape="$(study_auto_shape "$ranks")"
                study_case label="$backend ${n} ${spec}" backend="$backend" \
                    ranks="$ranks" threads="$threads" shape="$shape" \
                    grid="$n $n $n" steps="$steps" repeats="$repeats"
            done
        done
        echo
    done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo "=== risultato ==="
    awk -F, -v phase=05_hybrid '
    NR == 1 || $1 != phase || $32 != "ok" { next }
    {
        prod = $8 * $9
        k = $3 "," $10 "," prod "," $8
        wall[k] = $17; mpims[k] = $18; unt[k] = $27; thr[k] = $9
        gk = $10 "," prod
        if (!(gk in seen_g)) { gs[++ng] = gk; seen_g[gk] = 1 }
        if (!(gk "," $8 in seen_r)) {
            ranks[gk] = ranks[gk] " " $8; seen_r[gk "," $8] = 1
        }
    }
    END {
        for (i = 1; i <= ng; i++) {
            gk = gs[i]; split(gk, p, ",")
            printf "\n  %s^3, rank x thread = %s\n", p[1], p[2]
            printf "  %-12s %24s %24s\n", "", "schur", "pipeline"
            printf "  %-12s %10s %6s %6s %10s %6s %6s\n",
                   "rank x th", "ms/passo", "vs 1o", "%mpi",
                   "ms/passo", "vs 1o", "%mpi"
            n = split(ranks[gk], list, " ")
            # ordine crescente di rank
            for (a = 1; a <= n; a++) for (b = a + 1; b <= n; b++)
                if (list[a] + 0 > list[b] + 0) { t = list[a]; list[a] = list[b]; list[b] = t }
            base_a = ""; base_b = ""; min_a = 0; max_a = 0; min_b = 0; max_b = 0
            for (j = 1; j <= n; j++) {
                r = list[j]
                ka = "schur," gk "," r; kb = "pipeline," gk "," r
                lab = r " x " (ka in thr ? thr[ka] : thr[kb])
                if (base_a == "" && (ka in wall)) base_a = wall[ka]
                if (base_b == "" && (kb in wall)) base_b = wall[kb]
                printf "  %-12s", lab
                if (ka in wall) {
                    printf " %10.1f %5.2fx %5.0f%%", wall[ka], base_a/wall[ka], 100*mpims[ka]/wall[ka]
                    if (min_a == 0 || wall[ka] < min_a) min_a = wall[ka]
                    if (wall[ka] > max_a) max_a = wall[ka]
                } else printf " %10s %6s %6s", "-", "-", "-"
                if (kb in wall) {
                    printf " %10.1f %5.2fx %5.0f%%", wall[kb], base_b/wall[kb], 100*mpims[kb]/wall[kb]
                    if (min_b == 0 || wall[kb] < min_b) min_b = wall[kb]
                    if (wall[kb] > max_b) max_b = wall[kb]
                } else printf " %10s %6s %6s", "-", "-", "-"
                printf "\n"
            }
            if (min_a > 0)
                printf "    schur:    dal migliore al peggiore %.1fx\n", max_a / min_a
            if (min_b > 0)
                printf "    pipeline: dal migliore al peggiore %.1fx\n", max_b / min_b
        }
        print ""
        print "  Tutte le righe di un blocco usano lo stesso numero di core:"
        print "  `vs 1o\x27 e\x27 un rapporto, non uno speedup. Un rapporto sotto 1"
        print "  vuol dire che spezzare il dominio in quel modo costa."
        print ""
        print "  Il numero da guardare e\x27 l\x27ultimo di ogni blocco: se per schur"
        print "  e\x27 grande e per la pipeline no, il costo non e\x27 dividere il"
        print "  dominio -- e\x27 aprire un team di thread per ogni piano."
    }' "$STUDY_CSV"
fi

study_end
