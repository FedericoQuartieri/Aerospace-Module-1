#!/usr/bin/env bash
#PBS -N nsb-convergence
#PBS -q cpu
#PBS -l select=1:ncpus=28
#PBS -l walltime=04:00:00
#PBS -j oe
#
# Lo studio di convergenza alla griglia che serve davvero: 256^3.
#
#   qsub scripts/run_convergence_cluster.sh
#
# E' un involucro attorno a scripts/run_convergence.sh, che fa gia' tutto il
# lavoro: raffinamento spaziale a dt fisso, raffinamento temporale a griglia
# fissa, e il calcolo degli ordini osservati.
#
# ----------------------------------------------------------------------------
# Perche' sulla coda `cpu' e non su `scalability'
# ----------------------------------------------------------------------------
#
# Perche' questo studio misura ERRORI, non tempi. Un nodo condiviso coi job di
# altri utenti rallenta la corsa ma non sposta di una cifra le norme L2, quindi
# l'esclusivita' del nodo -- che su `scalability' costa il vincolo di mezz'ora
# -- qui non serve a niente. Su `cpu' ci sono 48 ore di walltime, e servono:
# lo studio temporale a 256^3 somma 300 passi temporali.
#
# Il rovescio della medaglia e' il tetto di 28 CPU per job su quella coda, cioe'
# 14 core fisici. Bastano: 14 core su 256^3 tengono il passo sotto il secondo.
#
# ----------------------------------------------------------------------------
# Perche' EXTRA_CFLAGS
# ----------------------------------------------------------------------------
#
# run_convergence.sh compila senza SIMD e senza thread, il che va benissimo per
# le griglie piccole in locale. A 256^3 per 160 passi non finirebbe dentro
# nessun walltime ragionevole, quindi qui gli si passano i flag dalla variabile
# d'ambiente che accetta apposta.
#
# I thread non cambiano il risultato: le linee sono indipendenti e ogni somma
# resta nell'ordine che aveva, quindi le norme sono identiche a quelle di un
# thread solo. Verificato a 32/64/128 in locale prima di scrivere questo.

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

build="build/convergence"
log="$build/cluster.log"
mkdir -p "$build"

exec > >(tee "$log") 2>&1

echo "=== macchina ==="
printf 'cpu concesse:  %s\n' "$(grep Cpus_allowed_list /proc/self/status | cut -f2)"

# Un thread per core FISICO concesso: l'SMT su questo carico non aggiunge
# unita' aritmetiche e in locale e' risultato controproducente.
allowed_cpus="$(grep -c ^processor /proc/cpuinfo)"
granted="$(grep Cpus_allowed_list /proc/self/status | cut -f2 |
           awk -F, '{n=0; for(i=1;i<=NF;i++){split($i,r,"-");
                     n += (r[2]=="") ? 1 : r[2]-r[1]+1} print n}')"
threads="${OMP_NUM_THREADS:-$(( granted / 2 ))}"
[[ "$threads" -lt 1 ]] && threads=1
printf 'thread usati:  %s\n' "$threads"
echo

echo "=== studio di convergenza a 256^3 ==="
echo "  spaziale: 32, 64, 128, 256 a dt fisso"
echo "  temporale: 256^3 con dt dimezzato quattro volte (300 passi in tutto)"
echo

export OMP_NUM_THREADS="$threads"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export EXTRA_CFLAGS="-mavx2 -mfma -DUSE_SIMD -fopenmp -DUSE_OMP"

./scripts/run_convergence.sh

echo
echo "log in $log"
