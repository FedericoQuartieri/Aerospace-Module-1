#include "utils.h"
#include "solver.h"
#include "parallel.h"
#include "workers.h"

void *xmalloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error allocating %zu bytes\n", size);
        exit(1);
    }
    return ptr;
}

void print_stats(const Decomp *d,
                 const SolverStats *solver_stats,
                 size_t sample_count) {
    /*
     * Il tempo che conta e' il piu' lento fra i processi: e' quello che
     * l'utente aspetta. Le riduzioni vanno fatte da tutti, quindi prima del
     * filtro sul rank, altrimenti gli altri resterebbero in attesa.
     */
    long long slowest_ns = par_max_long((long long)solver_stats->solve_steps);
    long long comm_ns = par_max_long((long long)par_comm_nanoseconds());

    /* Every process would otherwise print its own copy of the report. */
    if (par_rank() != 0) {
        return;
    }

    if (sample_count == 0) {
        printf("Solver time stats: no samples\n");
        return;
    }

    double ns_to_ms = 1.0e-6;
    double eta_avg_ms =
        (double)solver_stats->eta_sys * ns_to_ms / (double)sample_count;
    double zeta_avg_ms =
        (double)solver_stats->zeta_sys * ns_to_ms / (double)sample_count;
    double u_avg_ms =
        (double)solver_stats->u_sys * ns_to_ms / (double)sample_count;
    double psi_avg_ms =
        (double)solver_stats->psi_sys * ns_to_ms / (double)sample_count;
    double phi_low_avg_ms =
        (double)solver_stats->phi_low_sys * ns_to_ms /
        (double)sample_count;
    double phi_high_avg_ms =
        (double)solver_stats->phi_high_sys * ns_to_ms /
        (double)sample_count;
    double pressure_update_avg_ms =
        (double)solver_stats->pressure_update * ns_to_ms /
        (double)sample_count;
    double wr_output_avg_ms =
        (double)solver_stats->wr_output * ns_to_ms /
        (double)sample_count;
    double solve_steps_ns = (double)solver_stats->solve_steps;
    double percentage_factor =
        solve_steps_ns > 0.0 ? 100.0 / solve_steps_ns : 0.0;
    double solve_steps_avg_ns =
        (double)solver_stats->solve_steps / (double)sample_count;
    double local_cells =
        (double)d->n[0] * (double)d->n[1] * (double)d->n[2];
    double per_cell_step = (solve_steps_avg_ns / local_cells) / 10.0;
    printf("Grid: %d x %d x %d\n", d->n[0], d->n[1], d->n[2]);
    printf("Processes: %d\n", par_size());
    /* Ranghi e thread insieme: una misura senza entrambi i numeri non si sa
     * confrontare con nessun'altra. */
    printf("Threads per process: %d\n", workers_available());
    printf("Time steps: %zu\n", sample_count);
    printf("Solver time stats (average per time step):\n");
    printf("  eta system:  %.3f ms (%5.1f%%)\n", eta_avg_ms,
           (double)solver_stats->eta_sys * percentage_factor);
    printf("  zeta system: %.3f ms (%5.1f%%)\n", zeta_avg_ms,
           (double)solver_stats->zeta_sys * percentage_factor);
    printf("  u system:    %.3f ms (%5.1f%%)\n", u_avg_ms,
           (double)solver_stats->u_sys * percentage_factor);
    printf("  psi system:  %.3f ms (%5.1f%%)\n", psi_avg_ms,
           (double)solver_stats->psi_sys * percentage_factor);
    printf("  phi low:     %.3f ms (%5.1f%%)\n", phi_low_avg_ms,
           (double)solver_stats->phi_low_sys * percentage_factor);
    printf("  phi high:    %.3f ms (%5.1f%%)\n", phi_high_avg_ms,
           (double)solver_stats->phi_high_sys * percentage_factor);
    printf("  pressure:    %.3f ms (%5.1f%%)\n", pressure_update_avg_ms,
           (double)solver_stats->pressure_update * percentage_factor);
    printf("  write file:  %.3f ms\n", wr_output_avg_ms);
    /* Righe pensate per essere lette anche da uno script. */
    printf("  wall per step: %.3f ms\n",
           (double)slowest_ns * ns_to_ms / (double)sample_count);
    printf("  mpi per step:  %.3f ms (%5.1f%%)\n",
           (double)comm_ns * ns_to_ms / (double)sample_count,
           slowest_ns > 0 ? 100.0 * (double)comm_ns / (double)slowest_ns : 0.0);
    printf("  per cell-step: %.3f 1e-8s\n", per_cell_step);
}
