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

void print_stats(const SolverStats *stats, size_t sample_count) {
    if (sample_count == 0) {
        printf("Solver time stats: no samples\n");
        return;
    }

    const double ns_to_ms = 1.0e-6;
    const double percentage = stats->solve_steps > 0 
        ? 100.0 / (double)stats->solve_steps : 0.0;
#define AVG_MS(member) ((double)stats->member * ns_to_ms / (double)sample_count)
#define PERCENT(member) ((double)stats->member * percentage)

    double solve_steps_avg_ns = (double)stats->solve_steps / (double)sample_count;
    double per_cell_step = (solve_steps_avg_ns / GRID_CELLS) / 10.0;

    printf("Grid: %d x %d x %d\n", WIDTH, HEIGHT, DEPTH);
    printf("Time steps: %zu\n", sample_count);
    printf("Solver time stats (average per time step):\n");
    printf("  momentum halo: %.3f ms (%5.1f%%)\n",
           AVG_MS(momentum_halo), PERCENT(momentum_halo));
    printf("  eta system:    %.3f ms (%5.1f%%)\n",
           AVG_MS(eta_sys), PERCENT(eta_sys));
    printf("  zeta system:   %.3f ms (%5.1f%%)\n",
           AVG_MS(zeta_sys), PERCENT(zeta_sys));
    printf("  u system:      %.3f ms (%5.1f%%)\n",
           AVG_MS(u_sys), PERCENT(u_sys));
    printf("  pressure halo: %.3f ms (%5.1f%%)\n",
           AVG_MS(pressure_halo), PERCENT(pressure_halo));
    printf("  psi system:    %.3f ms (%5.1f%%)\n",
           AVG_MS(psi_sys), PERCENT(psi_sys));
    printf("  phi low:       %.3f ms (%5.1f%%)\n",
           AVG_MS(phi_low_sys), PERCENT(phi_low_sys));
    printf("  phi high:      %.3f ms (%5.1f%%)\n",
           AVG_MS(phi_high_sys), PERCENT(phi_high_sys));
    printf("  pressure:      %.3f ms (%5.1f%%)\n",
           AVG_MS(pressure_update), PERCENT(pressure_update));
    printf("  write file:    %.3f ms\n", AVG_MS(wr_output));
    printf("  per cell-step: %.3f 1e-8s\n", per_cell_step);

#undef AVG_MS
#undef PERCENT
}
