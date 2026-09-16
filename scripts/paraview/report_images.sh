#!/usr/bin/env bash
#
# The ParaView images of the report, from the runs of scripts/run_figures.sh.
#
#   ./scripts/paraview/report_images.sh
#   PVPYTHON=/opt/ParaView-5.12.1-MPI-Linux-Python3.10-x86_64/bin/pvpython \
#       ./scripts/paraview/report_images.sh
#
# Writes into report/figures/, replacing what is there: the mid-plane stills
# of make_still.py and the perspective views of make_3d.py.  The two stills
# of the moving sphere share the colour scale, so they can be compared.

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
pvpython="${PVPYTHON:-pvpython}"
out="$root/report/figures"
data="$root/data/output"

if ! command -v "$pvpython" > /dev/null; then
    echo "pvpython not found: set PVPYTHON to its path" >&2
    exit 1
fi
for case in cavity channel_obstacle moving_sphere; do
    if [[ ! -d "$data/$case" ]]; then
        echo "no $data/$case: run ./scripts/run_figures.sh $case first" >&2
        exit 1
    fi
done

still() { "$pvpython" "$root/scripts/paraview/make_still.py" "$@" 2>&1 | tail -n 1; }
view() { "$pvpython" "$root/scripts/paraview/make_3d.py" "$@" 2>&1 | tail -n 1; }

still "$data/cavity" "$out/cavity.png" --width 1200
still "$data/channel_obstacle" "$out/channel_obstacle.png" \
    --seeds inlet --streamlines 28
for frame in 10 20; do
    still "$data/moving_sphere" "$out/moving_sphere_t${frame}0.png" \
        --frame "$frame" --seeds inlet --streamlines 28 --range 0 2.5
done

# Half the page wide: larger labels.
view "$data/cavity" "$out/cavity_3d.png" \
    --sections 0.5 --seeds 0.5 0.5 0.85 0.6 0.9 --grid 3 2 \
    --elevation=30 --width 1400 --height 1100 --text 40
view "$data/channel_obstacle" "$out/channel_obstacle_3d.png" \
    --sections 0.25 0.75 --seeds 0.02 0.1 0.9 0.55 0.9 --grid 5 3 \
    --elevation=35 --width 1400 --height 800 --text 40
view "$data/moving_sphere" "$out/moving_sphere_3d.png" --frame 10 \
    --sections 0.677 --seeds 0.02 0.32 0.68 0.55 0.8 --grid 4 3 \
    --elevation=35 --width 1400 --height 800 --text 40
