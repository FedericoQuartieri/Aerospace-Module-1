#!/usr/bin/env bash
#PBS -N nsb-07-weak
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Fase 7 -- weak scaling: il lavoro per unita' di calcolo resta lo stesso e
# cresce il problema.
#
# Domanda: se ogni processo tiene sempre lo stesso blocco, il tempo per passo
# resta lo stesso?
#
# Per un codice a stencil e' la misura piu' onesta delle due. Lo strong scaling
# su una macchina sola satura la banda di memoria in fretta, e da li' in poi
# misura la banda invece dell'algoritmo; il weak scaling tiene la pressione per
# core costante e lascia vedere solo cio' che cresce davvero: la comunicazione
# e, per la pipeline, la lunghezza della catena.
#
# La seconda meta' della fase e' quella che nessuno ha ancora fatto: lo stesso
# weak scaling con i thread invece dei processi, sulle stesse identiche
# griglie. Stesso problema, stessa forma, una volta diviso fra 56 processi e
# una volta fra i 56 thread di un processo solo. La differenza fra le due
# colonne e' il prezzo netto di dividere il dominio, senza che nessuna delle
# due parti stia risolvendo un problema piu' piccolo dell'altra.
#
#   qsub scripts/study/07_weak.sh
#   PER_RANK=96 qsub scripts/study/07_weak.sh

source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"

cd "${PBS_O_WORKDIR:-$STUDY_ROOT}"
study_begin 07_weak
study_machine

PER_RANK="${PER_RANK:-64}"
UNITS="${UNITS:-1 2 4 7 8 14 28 56}"
STEPS="${STEPS:-10}"

CASE_SIMD=1
CASE_MPI=1
CASE_STEPS="$STEPS"
CASE_REPEATS="${REPEATS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-900}"

# La griglia globale di n unita' e' il blocco per unita' moltiplicato per la
# forma: cosi' ogni processo riceve esattamente PER_RANK^3 celle, che e' la
# definizione del weak scaling, e la forma resta quella che MPI sceglierebbe.
weak_grid()
{
    local shape="$1" px py pz
    read -r px py pz <<< "$shape"
    printf '%d %d %d' $(( px * PER_RANK )) $(( py * PER_RANK )) $(( pz * PER_RANK ))
}

echo "=== processi: ${PER_RANK}^3 celle ciascuno ==="
for backend in schur pipeline; do
    for u in $UNITS; do
        shape="$(study_auto_shape "$u")"
        [[ -z "$shape" ]] && continue
        study_case label="$backend R=$u" backend="$backend" omp=0 \
            ranks="$u" threads=1 shape="$shape" grid="$(weak_grid "$shape")"
    done
done
echo

echo "=== thread: le stesse griglie, un processo solo ==="
for backend in schur pipeline; do
    for u in $UNITS; do
        shape="$(study_auto_shape "$u")"
        [[ -z "$shape" ]] && continue
        study_case label="$backend T=$u" backend="$backend" omp=1 \
            ranks=1 threads="$u" shape="1 1 1" grid="$(weak_grid "$shape")"
    done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo
    echo "=== risultato ==="
    awk -F, -v phase=07_weak '
    NR == 1 || $1 != phase || $32 != "ok" { next }
    {
        units = ($8 > 1 ? $8 : $9)
        kind = ($8 > 1 ? "proc" : ($9 > 1 ? "thread" : "base"))
        cells = $10 * $11 * $12
        if (kind == "base") {
            wall["proc," $3 ",1"] = $17; wall["thread," $3 ",1"] = $17
            cell["proc," $3 ",1"] = cells; cell["thread," $3 ",1"] = cells
            mpip["proc," $3 ",1"] = $18; mpip["thread," $3 ",1"] = $18
            grid["proc," $3 ",1"] = $10 "x" $11 "x" $12
            grid["thread," $3 ",1"] = $10 "x" $11 "x" $12
            units = 1
        } else {
            wall[kind "," $3 "," units] = $17
            cell[kind "," $3 "," units] = cells
            mpip[kind "," $3 "," units] = $18
            grid[kind "," $3 "," units] = $10 "x" $11 "x" $12
        }
        if (!(units in seen)) { us[++nu] = units; seen[units] = 1 }
    }
    END {
        for (a = 1; a <= nu; a++) for (b = a + 1; b <= nu; b++)
            if (us[a] + 0 > us[b] + 0) { t = us[a]; us[a] = us[b]; us[b] = t }
        for (ki = 1; ki <= 2; ki++) {
            kind = (ki == 1 ? "proc" : "thread")
            printf "\n  %s\n", (kind == "proc" ? "processi" : "thread")
            printf "  %-6s %-16s %22s %22s\n", "", "", "schur", "pipeline"
            printf "  %-6s %-16s %10s %10s %10s %10s\n",
                   "unita", "griglia", "ms/passo", "effic.", "ms/passo", "effic."
            for (j = 1; j <= nu; j++) {
                u = us[j]
                ka = kind ",schur," u; kb = kind ",pipeline," u
                if (!(ka in wall) && !(kb in wall)) continue
                if (base_a[kind] == "" && (ka in wall)) base_a[kind] = wall[ka]
                if (base_b[kind] == "" && (kb in wall)) base_b[kind] = wall[kb]
                printf "  %-6s %-16s", u, (ka in grid ? grid[ka] : grid[kb])
                if (ka in wall) printf " %10.1f %9.0f%%", wall[ka], 100 * base_a[kind] / wall[ka]
                else printf " %10s %10s", "-", "-"
                if (kb in wall) printf " %10.1f %9.0f%%", wall[kb], 100 * base_b[kind] / wall[kb]
                else printf " %10s %10s", "-", "-"
                printf "\n"
            }
        }
        print ""
        print "  Efficienza = tempo a una unita\x27 / tempo a n unita\x27. Nel weak"
        print "  scaling il 100% e\x27 il risultato perfetto: stesso lavoro per"
        print "  unita\x27, stesso tempo. Quello che manca al 100% e\x27 comunicazione,"
        print "  sbilanciamento, o banda di memoria finita."
    }' "$STUDY_CSV"
fi

study_end
