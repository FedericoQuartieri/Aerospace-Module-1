#include "utils.h"

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

    printf("Solver time stats (average per time step):\n");
    printf("  eta system:  %.4f ms\n", eta_avg_ms);
    printf("  zeta system: %.4f ms\n", zeta_avg_ms);
    printf("  u system:    %.4f ms\n", u_avg_ms);
}
