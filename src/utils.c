#include "utils.h"
#include "solver.h"

void *xmalloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error allocating %zu bytes\n", size);
        exit(1);
    }
    return ptr;
}

void print_stats(const SolverStats *solver_stats, size_t sample_count) {
    if (sample_count == 0) {
        printf("Solver time stats: no samples\n");
        return;
    }

    const double ns_to_ms = 1.0e-6;
    const double eta_avg_ms =
        (double)solver_stats->eta_sys * ns_to_ms / (double)sample_count;
    const double zeta_avg_ms =
        (double)solver_stats->zeta_sys * ns_to_ms / (double)sample_count;
    const double u_avg_ms =
        (double)solver_stats->u_sys * ns_to_ms / (double)sample_count;
    const double psi_avg_ms =
        (double)solver_stats->psi_sys * ns_to_ms / (double)sample_count;
    const double phi_low_avg_ms =
        (double)solver_stats->phi_low_sys * ns_to_ms /
        (double)sample_count;
    const double phi_high_avg_ms =
        (double)solver_stats->phi_high_sys * ns_to_ms /
        (double)sample_count;
    const double pressure_update_avg_ms =
        (double)solver_stats->pressure_update * ns_to_ms /
        (double)sample_count;
    const double solve_steps_avg_ns =
        (double)solver_stats->solve_steps / (double)sample_count;
    double per_cell_step = (solve_steps_avg_ns / (GRID_CELLS))/10.0;
    printf("Solver time stats (average per time step):\n");
    printf("  eta system:  %.3f ms\n", eta_avg_ms);
    printf("  zeta system: %.3f ms\n", zeta_avg_ms);
    printf("  u system:    %.3f ms\n", u_avg_ms);
    printf("  psi system:  %.3f ms\n", psi_avg_ms);
    printf("  phi low:     %.3f ms\n", phi_low_avg_ms);
    printf("  phi high:    %.3f ms\n", phi_high_avg_ms);
    printf("  pressure:    %.3f ms\n", pressure_update_avg_ms);
    printf("  per cell-step: %.3f 1e-8s\n", per_cell_step);
}
