#!/usr/bin/env bash
#
# The three physical scenarios, run to make the figures of the report.
#
#   ./scripts/run_figures.sh                    # all three
#   ./scripts/run_figures.sh moving_sphere      # just one
#   RANKS=8 ./scripts/run_figures.sh
#
# Each case ends up in data/output/<case>/ (data/ is not tracked), the layout
# scripts/paraview/make_gif.py and open_series.py already expect.  A previous
# run of the same case is replaced.
#
# How many frames: the cavity and the fixed obstacle settle to a steady
# state, so only the first and the last step are written.  The moving sphere
# is an animation and writes every 10 steps, 21 frames.  A frame of the
# 192x96x96 channel is about 100 MB, of the 128^3 cavity about 120 MB.
#
# The tests do not read a configuration file: grid and steps are the ones
# compiled in (128^3 for the cavity, 192x96x96 for the channels, 200 steps to
# T = 1), and only the write frequency is set from here.

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ranks="${RANKS:-4}"
read -r -a mpirun <<< "${MPIRUN:-mpirun}"

declare -A scenario=(
    [cavity]="Lid-driven cavity"
    [channel_obstacle]="Channel with wall-attached inclined Brinkman obstacle"
    [moving_sphere]="Channel with moving spherical obstacle"
)
declare -A write_every=(
    [cavity]=200
    [channel_obstacle]=200
    [moving_sphere]=10
)

cases=("$@")
if [[ ${#cases[@]} -eq 0 ]]; then
    cases=(cavity channel_obstacle moving_sphere)
fi

for case in "${cases[@]}"; do
    if [[ -z "${scenario[$case]:-}" ]]; then
        echo "unknown case: $case (cavity, channel_obstacle, moving_sphere)" >&2
        exit 2
    fi
done

mkdir -p "$root/build/figures"

for case in "${cases[@]}"; do
    bin_dir="build/figures/$case"
    run_dir="$root/build/figures/run-$case"
    dest="$root/data/output/$case"

    printf '=== %s: %s processes, a frame every %s steps\n' \
        "$case" "$ranks" "${write_every[$case]}"

    if ! make -s -B --no-print-directory -C "$root" \
            MPI=1 SIMD=1 OMP=0 TEST_BIN_DIR="$bin_dir" \
            EXTRA_CPPFLAGS="-DDEFAULT_WR_FREQ=${write_every[$case]}" \
            "$bin_dir/$case" > "$root/build/figures/$case.build.log" 2>&1; then
        echo "build failed, see build/figures/$case.build.log" >&2
        exit 1
    fi

    rm -rf "$run_dir"
    mkdir -p "$run_dir"
    (cd "$run_dir" && "${mpirun[@]}" -n "$ranks" "$root/$bin_dir/$case") \
        | tee "$root/build/figures/$case.log" | tail -n 4

    rm -rf "$dest"
    mkdir -p "$(dirname "$dest")"
    mv "$run_dir/output/${scenario[$case]}" "$dest"
    rm -rf "$run_dir"
    printf '    %s frames in data/output/%s (%s)\n\n' \
        "$(find "$dest" -name 'sol_*.pvti' | wc -l)" "$case" \
        "$(du -sh "$dest" | cut -f1)"
done
