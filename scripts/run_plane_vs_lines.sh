#!/usr/bin/env bash
#PBS -N nsb-plane-lines
#PBS -q scalability
#PBS -l select=1:ncpus=112
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Confronto controllato delle due strutture OpenMP dei solver direzionali:
#
#   planes  un thread prende un piano e risolve tutte le sue linee;
#   lines   tutti avanzano sullo stesso piano e si spartiscono le linee;
#   serial  controllo: quei solver restano seriali, il resto conserva OpenMP.
#
# MPI e SIMD sono volutamente assenti. Con MPI i piani non sono una scelta
# lecita sugli assi distribuiti; con SIMD i passi Y e Z scavalcherebbero il
# codice sotto esame. I tre binari differiscono quindi per una sola -D.
#
# Sul cluster:
#   qsub scripts/run_plane_vs_lines.sh
#
# Prova locale rapida:
#   GRIDS='32x32x32 32x32x8' STEPS=2 REPEATS=2 THREADS='1 2 4' \
#       ./scripts/run_plane_vs_lines.sh
#
# Impostazioni principali sovrascrivibili dall'ambiente:
#   GRIDS='256x256x256 1024x512x32'
#   THREADS='7 14 28 56'
#   STEPS=3 REPEATS=3 BIND=spread WAIT_POLICY=active

set -euo pipefail
export LC_ALL=C

cd "${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

# Le due forme hanno entrambe 16 777 216 celle. La seconda lascia solo 32
# piani nei passi X/Y: confrontarla col cubo evita di concludere che una
# politica sia sempre migliore guardando una sola geometria favorevole.
GRIDS="${GRIDS:-256x256x256 1024x512x32}"
# I quattro punti coprono il regime utile del nodo e tengono il test entro i
# 30 minuti della coda. Per includere anche i costi a bassa concorrenza, sul
# cluster: export THREADS='1 2 4 7 14 28 56';
#          qsub -v THREADS scripts/run_plane_vs_lines.sh
THREADS="${THREADS:-7 14 28 56}"
STEPS="${STEPS:-3}"
REPEATS="${REPEATS:-3}"
VERIFY_GRID="${VERIFY_GRID:-32}"
VERIFY_STEPS="${VERIFY_STEPS:-3}"
VERIFY_THREADS="${VERIFY_THREADS:-56}"
PLACES="${PLACES:-cores}"
BIND="${BIND:-spread}"
WAIT_POLICY="${WAIT_POLICY:-active}"

build="build/plane-vs-lines"
raw="$build/results.csv"
summary="$build/summary.csv"
log="$build/run.log"
mkdir -p "$build"
# Se compilazione o correttezza falliscono, non lasciare in giro CSV di una
# misura precedente che potrebbero essere scambiati per il risultato corrente.
: > "$raw"
: > "$summary"
exec > >(tee "$log") 2>&1

die()
{
    printf 'ERRORE: %s\n' "$*" >&2
    exit 1
}

is_positive_integer()
{
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_positive_integer "$STEPS" || die "STEPS deve essere un intero positivo"
is_positive_integer "$REPEATS" || die "REPEATS deve essere un intero positivo"
is_positive_integer "$VERIFY_GRID" || die "VERIFY_GRID deve essere positivo"
is_positive_integer "$VERIFY_STEPS" || die "VERIFY_STEPS deve essere positivo"
is_positive_integer "$VERIFY_THREADS" || die "VERIFY_THREADS deve essere positivo"

# Conta le coppie socket/core, quindi non scambia i thread SMT per core fisici.
physical=$(lscpu -p=socket,core | awk -F, '!/^#/ {print $1 "," $2}' |
           sort -u | wc -l)
logical=$(lscpu -p=cpu | awk -F, '!/^#/ {n++} END {print n}')
allowed=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)
granted=$(awk -v list="$allowed" 'BEGIN {
    n = split(list, fields, ","); total = 0
    for (i = 1; i <= n; i++) {
        m = split(fields[i], edge, "-")
        total += (m == 1) ? 1 : edge[2] - edge[1] + 1
    }
    print total
}')

usable=$physical
[[ "$granted" -lt "$usable" ]] && usable=$granted

used_threads=()
for threads in $THREADS; do
    is_positive_integer "$threads" || die "THREADS contiene '$threads'"
    if [[ "$threads" -le "$usable" ]]; then
        used_threads+=("$threads")
    else
        printf 'salto %s thread: sono utilizzabili al massimo %s\n' \
            "$threads" "$usable"
    fi
done
[[ "${#used_threads[@]}" -gt 0 ]] || die "nessun valore THREADS utilizzabile"

if [[ "$VERIFY_THREADS" -gt "$usable" ]]; then
    VERIFY_THREADS=$usable
fi

compiler="${CC:-cc}"

echo "=== esperimento ==="
printf 'data:              %s\n' "$(date -Is)"
printf 'host:              %s\n' "$(hostname)"
printf 'commit:            %s\n' "$(git rev-parse --short HEAD 2>/dev/null || echo sconosciuto)"
printf 'compiler:          %s\n' "$($compiler --version | sed -n '1p')"
printf 'core fisici:       %s\n' "$physical"
printf 'cpu logiche:       %s\n' "$logical"
printf 'cpu concesse:      %s (%s)\n' "$allowed" "$granted"
printf 'thread provati:    %s\n' "${used_threads[*]}"
printf 'griglie:           %s\n' "$GRIDS"
printf 'passi/ripetizioni: %s / %s\n' "$STEPS" "$REPEATS"
printf 'OpenMP:            dynamic=false places=%s bind=%s wait=%s\n' \
    "$PLACES" "$BIND" "$WAIT_POLICY"
printf 'MPI:               spento (cc diretto, nessun mpirun)\n'
printf 'SIMD:              spento\n'

if [[ "$granted" -lt "$logical" ]]; then
    cat <<EOF

ATTENZIONE: il processo vede $granted CPU logiche sulle $logical della macchina.
La misura e' valida soltanto se queste CPU sono state assegnate in esclusiva.
Il job PBS in testa allo script richiede apposta il nodo intero.
EOF
fi

if [[ -n "$(git status --short 2>/dev/null)" ]]; then
    echo
    echo "worktree non pulito (registrato per la riproducibilita'):"
    git status --short
fi
echo

core_sources=()
for source in src/*.c; do
    [[ "$(basename -- "$source")" == "main.c" ]] || core_sources+=("$source")
done

common_cflags=(-std=gnu11 -O3 -Wall -Wextra -Iinclude
               -fopenmp -DUSE_OMP)

policy_macro()
{
    case "$1" in
        planes) printf '%s' WORKERS_LINE_POLICY_PLANES ;;
        lines)  printf '%s' WORKERS_LINE_POLICY_LINES ;;
        serial) printf '%s' WORKERS_LINE_POLICY_SERIAL ;;
        *) die "politica sconosciuta '$1'" ;;
    esac
}

compile_policy()
{
    local policy="$1"
    local define="-DWORKERS_LINE_POLICY=$(policy_macro "$policy")"

    printf '  %-7s %s\n' "$policy" "$define"
    "$compiler" "${common_cflags[@]}" "$define" \
        src/main.c "${core_sources[@]}" -fopenmp -lm \
        -o "$build/solver-$policy"

    "$compiler" "${common_cflags[@]}" "$define" \
        -DDEFAULT_WIDTH="$VERIFY_GRID" \
        -DDEFAULT_HEIGHT="$VERIFY_GRID" \
        -DDEFAULT_DEPTH="$VERIFY_GRID" \
        -DDEFAULT_STEPS="$VERIFY_STEPS" \
        test/paper_man.c "${core_sources[@]}" -fopenmp -lm \
        -o "$build/verify-$policy"
}

echo "=== compilazione: cambia soltanto WORKERS_LINE_POLICY ==="
for policy in planes lines serial; do
    compile_policy "$policy"
done

if command -v ldd >/dev/null; then
    for policy in planes lines serial; do
        if ldd "$build/solver-$policy" 2>/dev/null |
           grep -Eq 'libmpi|libopen-rte|libopen-pal'; then
            die "$build/solver-$policy e' collegato a MPI"
        fi
    done
fi
echo

run_with_openmp()
{
    local threads="$1"
    shift
    env OMP_NUM_THREADS="$threads" \
        OMP_THREAD_LIMIT="$threads" \
        OMP_DYNAMIC=false \
        OMP_PLACES="$PLACES" \
        OMP_PROC_BIND="$BIND" \
        OMP_WAIT_POLICY="$WAIT_POLICY" \
        "$@"
}

assert_run_identity()
{
    local output="$1" expected_policy="$2" expected_threads="$3"
    local got_policy got_processes got_threads got_mpi

    got_policy=$(awk '/^Directional policy:/ {print $3}' "$output")
    got_processes=$(awk '/^Processes:/ {print $2}' "$output")
    got_threads=$(awk '/^Threads per process:/ {print $4}' "$output")
    got_mpi=$(awk '/^  mpi per step:/ {print $4}' "$output")

    [[ "$got_policy" == "$expected_policy" ]] ||
        die "$output dichiara policy '$got_policy', attesa '$expected_policy'"
    [[ "$got_processes" == 1 ]] ||
        die "$output ha Processes=$got_processes: il test deve essere seriale MPI"
    [[ "$got_threads" == "$expected_threads" ]] ||
        die "$output ha $got_threads thread, attesi $expected_threads"
    [[ "$got_mpi" =~ ^0([.]0+)?$ ]] ||
        die "$output contiene mpi per step = $got_mpi"
}

echo "=== correttezza prima delle prestazioni ==="
for policy in planes lines serial; do
    output="$build/verify-$policy.out"
    run_with_openmp "$VERIFY_THREADS" "$build/verify-$policy" > "$output"
    assert_run_identity "$output" "$policy" "$VERIFY_THREADS"
    awk '/^  L2 error/ {print}' "$output" > "$build/verify-$policy.l2"
    [[ "$(wc -l < "$build/verify-$policy.l2")" -eq 4 ]] ||
        die "mancano le quattro norme L2 nell'output $policy"
done

for policy in lines serial; do
    if ! cmp -s "$build/verify-planes.l2" "$build/verify-$policy.l2"; then
        echo "Le norme di planes e $policy sono diverse:" >&2
        diff -u "$build/verify-planes.l2" "$build/verify-$policy.l2" || true
        exit 1
    fi
done
printf 'PASS: planes, lines e serial producono le stesse quattro norme L2 '\
'(%s^3, %s thread).\n\n' "$VERIFY_GRID" "$VERIFY_THREADS"

printf '%s\n' \
    'grid,policy,binding,threads,repeat,wall_ms,directional_ms,eta_ms,zeta_ms,u_ms,psi_ms,phi_low_ms,phi_high_ms,pressure_update_ms,porosity_ms,unaccounted_ms,mpi_ms' \
    > "$raw"

parse_times()
{
    awk '
    /^  eta system:/   {eta = $3; have_eta = 1}
    /^  zeta system:/  {zeta = $3; have_zeta = 1}
    /^  u system:/     {u = $3; have_u = 1}
    /^  psi system:/   {psi = $3; have_psi = 1}
    /^  phi low:/      {low = $3; have_low = 1}
    /^  phi high:/     {high = $3; have_high = 1}
    /^  pressure:/     {pressure = $2; have_pressure = 1}
    /^  porosity:/     {porosity = $2; have_porosity = 1}
    /^  non contato:/  {unaccounted = $3; have_unaccounted = 1}
    /^  wall per step:/{wall = $4; have_wall = 1}
    /^  mpi per step:/ {mpi = $4; have_mpi = 1}
    END {
        complete = have_eta && have_zeta && have_u && have_psi &&
                   have_low && have_high && have_pressure && have_porosity &&
                   have_unaccounted && have_wall && have_mpi
        if (!complete) exit 2
        directional = eta + zeta + u + psi + low + high
        printf "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
               wall, directional, eta, zeta, u, psi, low, high,
               pressure, porosity, unaccounted, mpi
    }' "$1"
}

measure()
{
    local grid="$1" policy="$2" threads="$3" repeat="$4" config="$5"
    local output="$build/${grid}-${policy}-t${threads}-r${repeat}.out"
    local times

    run_with_openmp "$threads" "$build/solver-$policy" "$config" > "$output"
    assert_run_identity "$output" "$policy" "$threads"
    times=$(parse_times "$output") ||
        die "non riesco a leggere tutti i tempi da $output"

    printf '%s,%s,%s,%s,%s,%s\n' \
        "$grid" "$policy" "$BIND" "$threads" "$repeat" "$times" >> "$raw"
    printf '    r=%s %-7s %9s ms\n' "$repeat" "$policy" "${times%%,*}"
}

echo "=== misure ==="
for grid in $GRIDS; do
    IFS=x read -r nx ny nz extra <<< "$grid"
    [[ -z "${extra:-}" ]] || die "griglia non valida '$grid'"
    is_positive_integer "${nx:-}" || die "griglia non valida '$grid'"
    is_positive_integer "${ny:-}" || die "griglia non valida '$grid'"
    is_positive_integer "${nz:-}" || die "griglia non valida '$grid'"

    config="$build/$grid.conf"
    printf 'width = %s\nheight = %s\ndepth = %s\nsteps = %s\nt_end = 1e-1\n' \
        "$nx" "$ny" "$nz" "$STEPS" > "$config"

    echo "  griglia $grid"
    for threads in "${used_threads[@]}"; do
        echo "  $threads thread"
        for ((repeat = 1; repeat <= REPEATS; repeat++)); do
            # Ruota l'ordine per non regalare sempre cache o temperatura alla
            # stessa politica.
            case $(( repeat % 3 )) in
                1) order=(planes lines serial) ;;
                2) order=(lines serial planes) ;;
                0) order=(serial planes lines) ;;
            esac
            for policy in "${order[@]}"; do
                measure "$grid" "$policy" "$threads" "$repeat" "$config"
            done
        done
    done
    echo
done

median_for()
{
    local grid="$1" policy="$2" threads="$3" column="$4"
    awk -F, -v grid="$grid" -v policy="$policy" -v threads="$threads" \
        -v column="$column" \
        'NR > 1 && $1 == grid && $2 == policy && $4 == threads {
             print $column
         }' "$raw" | sort -n | awk '
        {value[NR] = $1}
        END {
            if (NR == 0) exit 1
            if (NR % 2) median = value[(NR + 1) / 2]
            else        median = (value[NR / 2] + value[NR / 2 + 1]) / 2
            printf "%.3f", median
        }'
}

median_sum_for()
{
    local grid="$1" policy="$2" threads="$3" column_a="$4" column_b="$5"
    awk -F, -v grid="$grid" -v policy="$policy" -v threads="$threads" \
        -v column_a="$column_a" -v column_b="$column_b" \
        'NR > 1 && $1 == grid && $2 == policy && $4 == threads {
             print $column_a + $column_b
         }' "$raw" | sort -n | awk '
        {value[NR] = $1}
        END {
            if (NR == 0) exit 1
            if (NR % 2) median = value[(NR + 1) / 2]
            else        median = (value[NR / 2] + value[NR / 2 + 1]) / 2
            printf "%.3f", median
        }'
}

ratio()
{
    awk -v numerator="$1" -v denominator="$2" \
        'BEGIN {printf "%.3f", numerator / denominator}'
}

printf '%s\n' \
    'grid,threads,planes_directional_ms,lines_directional_ms,serial_directional_ms,lines_over_planes,serial_over_lines,planes_wall_ms,lines_wall_ms,serial_wall_ms,wall_lines_over_planes,wall_serial_over_lines,x_lines_over_planes,y_lines_over_planes,z_lines_over_planes' \
    > "$summary"

for grid in $GRIDS; do
    for threads in "${used_threads[@]}"; do
        planes_dir=$(median_for "$grid" planes "$threads" 7)
        lines_dir=$(median_for "$grid" lines "$threads" 7)
        serial_dir=$(median_for "$grid" serial "$threads" 7)
        planes_wall=$(median_for "$grid" planes "$threads" 6)
        lines_wall=$(median_for "$grid" lines "$threads" 6)
        serial_wall=$(median_for "$grid" serial "$threads" 6)
        planes_x=$(median_sum_for "$grid" planes "$threads" 8 11)
        lines_x=$(median_sum_for "$grid" lines "$threads" 8 11)
        planes_y=$(median_sum_for "$grid" planes "$threads" 9 12)
        lines_y=$(median_sum_for "$grid" lines "$threads" 9 12)
        planes_z=$(median_sum_for "$grid" planes "$threads" 10 13)
        lines_z=$(median_sum_for "$grid" lines "$threads" 10 13)

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$grid" "$threads" \
            "$planes_dir" "$lines_dir" "$serial_dir" \
            "$(ratio "$lines_dir" "$planes_dir")" \
            "$(ratio "$serial_dir" "$lines_dir")" \
            "$planes_wall" "$lines_wall" "$serial_wall" \
            "$(ratio "$lines_wall" "$planes_wall")" \
            "$(ratio "$serial_wall" "$lines_wall")" \
            "$(ratio "$lines_x" "$planes_x")" \
            "$(ratio "$lines_y" "$planes_y")" \
            "$(ratio "$lines_z" "$planes_z")" >> "$summary"
    done
done

echo "=== risultato: mediana dei solver direzionali ==="
awk -F, '
NR == 1 {
    printf "  %-14s %6s %10s %10s %10s %8s %8s\n",
           "griglia", "thread", "planes", "lines", "serial", "L/P", "S/L"
    next
}
{
    printf "  %-14s %6s %10.3f %10.3f %10.3f %7.3fx %7.3fx\n",
           $1, $2, $3, $4, $5, $6, $7
}' "$summary"

echo
echo "=== lines / planes per asse ==="
awk -F, '
NR == 1 {
    printf "  %-14s %6s %9s %9s %9s\n",
           "griglia", "thread", "X", "Y", "Z"
    next
}
{
    printf "  %-14s %6s %8.3fx %8.3fx %8.3fx\n",
           $1, $2, $13, $14, $15
}' "$summary"

cat <<'EOF'

Lettura delle due colonne decisive:
  L/P > 1  -> planes e' piu' veloce di lines.
  L/P < 1  -> lines e' piu' veloce di planes.
  S/L > 1  -> lines porta un guadagno rispetto al solver direzionale seriale.
  S/L <= 1 -> il ramo lines non sta ripagando la propria complessita'.

Il verdetto primario usa la somma eta+zeta+u+psi+phi_low+phi_high; il wall
completo e gli stadi separati restano nel CSV. La seconda tabella applica lo
stesso rapporto separatamente a X, Y e Z: anche li', sopra 1 vince planes.
Rapporti vicini a 1 vanno considerati rumore e ripetuti, non trasformati in
una conclusione.

Questo test non autorizza planes con un dominio MPI distribuito: li' l'ordine
delle collettive resta un vincolo di correttezza. Misura se, quando entrambe le
strutture sono lecite, lines conviene e se batte almeno il controllo seriale.
EOF

printf 'dati grezzi: %s\n' "$raw"
printf 'mediane:     %s\n' "$summary"
printf 'log:         %s\n' "$log"
