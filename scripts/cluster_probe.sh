#!/usr/bin/env bash
#
# Cosa offre davvero un nodo di questo cluster.
#
# Lo studio di scaling ha senso solo se si sa contro cosa si sta misurando: il
# numero di core fisici (non quelli logici, che sono il doppio e non portano
# banda), quanti nodi NUMA ci sono, e soprattutto quanta banda di memoria il
# nodo riesce a dare. Quest'ultima e' il tetto vero: il solutore fa circa
# 0.1 flop per byte letto, quindi il suo limite non e' il calcolo, e sapere il
# numero della banda permette di dire se una misura e' buona o cattiva invece
# di limitarsi a riportarla.
#
# La banda viene misurata due volte, con e senza inizializzazione parallela.
# La differenza fra le due e' l'effetto del first-touch: le pagine appartengono
# al nodo NUMA del thread che le tocca per primo, quindi inizializzare in serie
# le concentra tutte su un socket e i thread dell'altro leggono in remoto.
#
# Uso, dentro un job PBS (OpenPBS 23):
#
#   qsub -I -l select=1:ncpus=<core> -l place=excl -l walltime=00:30:00
#   ./scripts/cluster_probe.sh
#
# place=excl non e' un dettaglio: senza, un altro job puo' finire sullo stesso
# nodo e consumare la banda che stiamo misurando.

set -euo pipefail

triad_elements="${TRIAD_ELEMENTS:-60000000}"
triad_repeats="${TRIAD_REPEATS:-5}"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

rule() { printf '\n=== %s %s\n' "$1" "$(printf '=%.0s' $(seq 1 $((66 - ${#1}))))"; }
field() { printf '  %-26s %s\n' "$1" "$2"; }

# ---------------------------------------------------------------- allocazione

rule "Allocazione PBS"

if [[ -n "${PBS_JOBID:-}" ]]; then
    field "job id" "$PBS_JOBID"
    field "coda" "${PBS_QUEUE:-?}"
    if [[ -r "${PBS_NODEFILE:-/dev/null}" ]]; then
        field "slot assegnati" "$(wc -l < "$PBS_NODEFILE")"
        field "nodi distinti" "$(sort -u "$PBS_NODEFILE" | wc -l)"
        field "nodi" "$(sort -u "$PBS_NODEFILE" | tr '\n' ' ')"
    fi
else
    field "attenzione" "fuori da un job PBS: i numeri sono quelli di questa macchina"
fi
field "host" "$(hostname)"

# ------------------------------------------------------------------------ cpu

rule "CPU"

lscpu_value() { lscpu | awk -F: -v k="$1" '$1 == k {gsub(/^ +/, "", $2); print $2; exit}'; }

sockets="$(lscpu_value 'Socket(s)')"
cores_per_socket="$(lscpu_value 'Core(s) per socket')"
threads_per_core="$(lscpu_value 'Thread(s) per core')"
logical="$(nproc)"
physical=$((sockets * cores_per_socket))

field "modello" "$(lscpu_value 'Model name')"
field "socket" "$sockets"
field "core per socket" "$cores_per_socket"
field "thread per core" "$threads_per_core"
field "core fisici" "$physical"
field "core logici (nproc)" "$logical"

if [[ "$threads_per_core" -gt 1 ]]; then
    printf '\n  L'\''hyperthreading e'\'' acceso: nproc dice %s ma i core veri sono %s.\n' \
        "$logical" "$physical"
    printf '  Per un codice memory bound i due thread di uno stesso core si\n'
    printf '  contendono le stesse unita'\'' load/store: vanno misurati una volta\n'
    printf '  (%s thread contro %s) e poi si lavora a %s.\n' \
        "$physical" "$logical" "$physical"
fi

rule "Cache"
for index in /sys/devices/system/cpu/cpu0/cache/index*; do
    [[ -r "$index/level" ]] || continue
    field "L$(cat "$index/level") $(cat "$index/type")" \
          "$(cat "$index/size") (condivisa da $(cat "$index/shared_cpu_list"))"
done

# ----------------------------------------------------------------------- numa

rule "NUMA"

if command -v numactl > /dev/null; then
    numactl -H | sed 's/^/  /'
else
    nodes="$(find /sys/devices/system/node -maxdepth 1 -name 'node[0-9]*' | wc -l)"
    field "nodi NUMA" "$nodes"
    for node in /sys/devices/system/node/node[0-9]*; do
        field "$(basename "$node") cpu" "$(cat "$node/cpulist")"
        field "$(basename "$node") mem" \
              "$(awk '/MemTotal/ {print $4, $5}' "$node/meminfo")"
    done
fi

rule "Memoria"
field "totale" "$(awk '/MemTotal/ {printf "%.1f GB", $2/1048576}' /proc/meminfo)"
field "libera" "$(awk '/MemAvailable/ {printf "%.1f GB", $2/1048576}' /proc/meminfo)"

# ------------------------------------------------------------------ toolchain

rule "Toolchain"
field "cc" "$(cc --version 2>/dev/null | head -1)"
if command -v mpicc > /dev/null; then
    field "mpicc" "$(mpicc -show 2>/dev/null | head -1)"
    field "mpirun" "$(mpirun --version 2>&1 | head -1)"
else
    field "mpicc" "assente: carica il modulo MPI prima di misurare"
fi

# ---------------------------------------------------------------------- banda

rule "Banda di memoria (triad)"

cat > "$workdir/triad.c" <<'EOF'
/*
 * Triad di STREAM: a[i] = b[i] + s * c[i].
 *
 * Due flussi letti e uno scritto, cioe' 24 byte per elemento contro 2 flop:
 * intensita' aritmetica 0.083 flop/byte. E' la stessa zona in cui vive
 * l'algoritmo di Thomas, quindi il numero che esce di qui e' il tetto contro
 * cui confrontare il solutore, non una curiosita' separata.
 *
 * argv[1] elementi per array, argv[2] 1 se l'inizializzazione va fatta in
 * parallelo (first touch distribuito) e 0 se va fatta in serie.
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    size_t n = (argc > 1) ? strtoull(argv[1], NULL, 10) : 60000000ull;
    int parallel_touch = (argc > 2) ? atoi(argv[2]) : 1;
    int repeats = (argc > 3) ? atoi(argv[3]) : 5;

    double *a = malloc(n * sizeof *a);
    double *b = malloc(n * sizeof *b);
    double *c = malloc(n * sizeof *c);

    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "memoria insufficiente per %zu elementi\n", n);
        return 1;
    }

    /*
     * malloc non tocca niente: e' questo ciclo che decide su quale nodo NUMA
     * finisce ogni pagina. Farlo in serie le concentra tutte sul socket del
     * thread principale, ed e' esattamente l'errore che si vuole misurare.
     */
    if (parallel_touch) {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
            a[i] = 0.0;
            b[i] = 1.0;
            c[i] = 2.0;
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            a[i] = 0.0;
            b[i] = 1.0;
            c[i] = 2.0;
        }
    }

    const double scalar = 3.0;
    double best = 1.0e30;

    for (int r = 0; r < repeats; r++) {
        double start = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i] + scalar * c[i];
        }
        double elapsed = omp_get_wtime() - start;
        if (elapsed < best) {
            best = elapsed;
        }
    }

    /* Somma finta: impedisce al compilatore di buttare via il ciclo. */
    double checksum = a[0] + a[n / 2] + a[n - 1];

    printf("%d %.2f %.3e\n",
           omp_get_max_threads(),
           3.0 * (double)n * sizeof(double) / best / 1.0e9,
           checksum);

    free(c);
    free(b);
    free(a);
    return 0;
}
EOF

arch_flag=""
if cc -march=native -E - < /dev/null > /dev/null 2>&1; then
    arch_flag="-march=native"
fi

if ! cc -O3 -fopenmp $arch_flag "$workdir/triad.c" -o "$workdir/triad" 2> "$workdir/cc.log"; then
    printf '  compilazione fallita:\n'
    sed 's/^/    /' "$workdir/cc.log"
    exit 1
fi

gib="$(awk -v n="$triad_elements" 'BEGIN {printf "%.1f", 3 * n * 8 / 1073741824}')"
printf '  %s elementi per array, %s GB in totale, %s ripetizioni, si tiene la migliore\n\n' \
    "$triad_elements" "$gib" "$triad_repeats"
printf '  %-9s %-14s %-14s %s\n' thread "GB/s (par.)" "GB/s (serie)" "perdita NUMA"

threads=1
while [[ "$threads" -le "$physical" ]]; do
    par="$(OMP_NUM_THREADS=$threads OMP_PROC_BIND=close OMP_PLACES=cores \
           "$workdir/triad" "$triad_elements" 1 "$triad_repeats" | awk '{print $2}')"
    ser="$(OMP_NUM_THREADS=$threads OMP_PROC_BIND=close OMP_PLACES=cores \
           "$workdir/triad" "$triad_elements" 0 "$triad_repeats" | awk '{print $2}')"
    loss="$(awk -v p="$par" -v s="$ser" \
            'BEGIN {printf "%.2fx", (s > 0) ? p / s : 0}')"
    printf '  %-9s %-14s %-14s %s\n' "$threads" "$par" "$ser" "$loss"
    threads=$((threads * 2))
done

# L'ultimo punto interessante e' con tutti i core logici: dice se
# l'hyperthreading porta banda (di solito no) o solo contesa.
if [[ "$logical" -ne "$physical" ]]; then
    par="$(OMP_NUM_THREADS=$logical OMP_PROC_BIND=spread OMP_PLACES=threads \
           "$workdir/triad" "$triad_elements" 1 "$triad_repeats" | awk '{print $2}')"
    printf '  %-9s %-14s %-14s %s\n' "$logical*" "$par" "-" "(hyperthread)"
fi

printf '\n  La colonna "perdita NUMA" e'\'' quanto si guadagna a inizializzare in\n'
printf '  parallelo. Sopra 1.2x conviene rifare il first-touch dei campi del\n'
printf '  solutore con lo stesso schedule(static) dei cicli di calcolo.\n'
printf '\n  Il massimo della colonna parallela e'\'' il tetto del roofline: il\n'
printf '  solutore non puo'\'' andare piu'\'' veloce di\n'
printf '  (banda) x (0.1 flop/byte), qualunque cosa si ottimizzi.\n'
