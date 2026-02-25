#!/bin/bash
# ==============================================================================
# Convergence Test Script
# Runs the manufactured solution test at multiple grid resolutions
# and collects L2 errors for convergence analysis
# ==============================================================================

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
RESULTS_FILE="${BUILD_DIR}/convergence_results.txt"
CONSTANTS_FILE="${PROJECT_ROOT}/include/constants.h"
CONSTANTS_BACKUP="${PROJECT_ROOT}/include/constants.h.backup"

# Grid sizes to test (must be > 2 for proper discretization)
GRID_SIZES=(8 16)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  Convergence Test for Navier-Stokes Solver"
echo "=============================================="
echo ""

# Backup original constants.h
cp "${CONSTANTS_FILE}" "${CONSTANTS_BACKUP}"
echo "Backed up constants.h"

# Clean results file
echo "# N dx L2_vx L2_vy L2_vz L2_p" > "${RESULTS_FILE}"

# Function to restore constants.h on exit
cleanup() {
    echo ""
    echo "Restoring original constants.h..."
    cp "${CONSTANTS_BACKUP}" "${CONSTANTS_FILE}"
    rm -f "${CONSTANTS_BACKUP}"
}
trap cleanup EXIT

# Function to modify constants.h with new grid size
modify_constants() {
    local N=$1
    local DT_NEW=$2
    
    sed -i.tmp \
        -e "s/^#define WIDTH .*/#define WIDTH ${N}/" \
        -e "s/^#define HEIGHT .*/#define HEIGHT ${N}/" \
        -e "s/^#define DEPTH .*/#define DEPTH ${N}/" \
        -e "s/^#define DT .*/#define DT ${DT_NEW}/" \
        "${CONSTANTS_FILE}"
    rm -f "${CONSTANTS_FILE}.tmp"
}

# Run test for each grid size
for N in "${GRID_SIZES[@]}"; do
    echo ""
    echo -e "${YELLOW}======================================${NC}"
    echo -e "${YELLOW}Testing with grid size N = ${N}${NC}"
    echo -e "${YELLOW}======================================${NC}"
    
    # Compute appropriate time step (CFL condition: dt ~ dx^2 for stability)
    # dx = 1/(N-1), dt should scale with dx^2 for diffusive terms
    DX=$(echo "scale=10; 1.0 / ($N - 1)" | bc)
    # Use dt = 0.1 * dx^2 / nu for stability (nu = 0.7)
    DT=$(echo "scale=10; 0.1 * $DX * $DX / 0.7" | bc)
    # Ensure minimum time step for reasonable simulation
    DT=$(echo "if ($DT < 0.0001) 0.0001 else $DT" | bc)
    
    echo "Grid spacing dx = ${DX}"
    echo "Time step dt = ${DT}"
    
    # Modify constants.h
    modify_constants ${N} ${DT}
    
    # Rebuild
    echo "Rebuilding..."
    cd "${BUILD_DIR}"
    make clean
    make test_manufactured -j4 2>&1 | tail -5
    
    # Run test and capture output
    echo "Running test_manufactured..."
    OUTPUT=$(./test_manufactured 2>&1 || true)
    
    # Parse errors from output (looking for L2 errors)
    # Expected format from print_test_result:
    #   v_x: L1=..., L2=..., Linf=...
    L2_VX=$(echo "$OUTPUT" | grep "v_x:" | sed -n 's/.*L2=\([^,]*\).*/\1/p')
    L2_VY=$(echo "$OUTPUT" | grep "v_y:" | sed -n 's/.*L2=\([^,]*\).*/\1/p')
    L2_VZ=$(echo "$OUTPUT" | grep "v_z:" | sed -n 's/.*L2=\([^,]*\).*/\1/p')
    L2_P=$(echo "$OUTPUT" | grep "p:" | sed -n 's/.*L2=\([^,]*\).*/\1/p')
    
    if [[ -z "$L2_VX" ]]; then
        echo -e "${RED}Failed to parse results for N=${N}${NC}"
        echo "Output was:"
        echo "$OUTPUT"
        L2_VX="nan"
        L2_VY="nan"
        L2_VZ="nan"
        L2_P="nan"
    else
        echo -e "${GREEN}L2 errors: vx=${L2_VX}, vy=${L2_VY}, vz=${L2_VZ}, p=${L2_P}${NC}"
    fi
    
    # Save results
    echo "${N} ${DX} ${L2_VX} ${L2_VY} ${L2_VZ} ${L2_P}" >> "${RESULTS_FILE}"
done

echo ""
echo "=============================================="
echo "  Convergence Results"
echo "=============================================="
cat "${RESULTS_FILE}"

echo ""
echo "Results saved to: ${RESULTS_FILE}"
echo ""

# Compute convergence rates
echo "=============================================="
echo "  Convergence Rate Analysis"
echo "=============================================="

# Read results and compute rates
LC_NUMERIC=C awk 'BEGIN {
    prev_N = 0; prev_dx = 0; prev_L2 = 0;
    print "N\t\tdx\t\tL2(vx)\t\tRate"
    print "---\t\t---\t\t---\t\t---"
}
!/^#/ {
    N = $1; dx = $2; L2 = $3;
    if (prev_N > 0 && L2 != "nan" && prev_L2 != "nan") {
        rate = log(prev_L2 / L2) / log(prev_dx / dx);
        printf "%d\t\t%.4e\t%.4e\t%.2f\n", N, dx, L2, rate;
    } else if (L2 != "nan") {
        printf "%d\t\t%.4e\t%.4e\t-\n", N, dx, L2;
    }
    prev_N = N; prev_dx = dx; prev_L2 = L2;
}' "${RESULTS_FILE}"

echo ""
echo "Expected rate for 2nd order method: ~2.0"
echo "=============================================="
