#include "utils.h"
#include "backend.h"
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

static int report_pipeline_batch_lines(void) {
#if defined(PIPELINE_BATCH_LINES)
    return PIPELINE_BATCH_LINES;
#else
    return 0;
#endif
}

void print_stats(const Decomp *d,
                 const SolverStats *solver_stats,
                 size_t sample_count) {
    /*
     * Il tempo che conta e' il piu' lento fra i processi: e' quello che
     * l'utente aspetta. Le riduzioni vanno fatte da tutti, quindi prima del
     * filtro sul rank, altrimenti gli altri resterebbero in attesa.
     */
    const long long local_solve_steps =
        (long long)solver_stats->solve_steps;
    const long long local_timed_stages =
        (long long)solver_stats->eta_sys +
        (long long)solver_stats->zeta_sys +
        (long long)solver_stats->u_sys +
        (long long)solver_stats->psi_sys +
        (long long)solver_stats->phi_low_sys +
        (long long)solver_stats->phi_high_sys +
        (long long)solver_stats->pressure_update +
        (long long)solver_stats->porosity_fill;
    const long long local_comm_ns =
        (long long)par_comm_nanoseconds();
    const long long local_cells =
        (long long)d->n[0] * (long long)d->n[1] * (long long)d->n[2];

    long long slowest_ns = par_max_long(local_solve_steps);
    int slowest_rank = par_rank_of_max_long(local_solve_steps);
    long long eta_ns = par_max_long((long long)solver_stats->eta_sys);
    long long zeta_ns = par_max_long((long long)solver_stats->zeta_sys);
    long long u_ns = par_max_long((long long)solver_stats->u_sys);
    long long psi_ns = par_max_long((long long)solver_stats->psi_sys);
    long long phi_low_ns = par_max_long((long long)solver_stats->phi_low_sys);
    long long phi_high_ns =
        par_max_long((long long)solver_stats->phi_high_sys);
    long long pressure_update_ns =
        par_max_long((long long)solver_stats->pressure_update);
    long long porosity_fill_ns =
        par_max_long((long long)solver_stats->porosity_fill);
    /*
     * Il termine noto del passo eta. Sta DENTRO eta_sys, non accanto: non va
     * sommato agli stadi, o il passo risulterebbe piu' lungo di quello che e'.
     * Si legge per differenza -- eta meno questo e' il solutore -- ed e' la
     * differenza a dire se conviene lavorare sul termine noto o sul sistema.
     */
    long long g_term_ns =
        par_max_long((long long)solver_stats->momentum_source);
    long long wr_output_ns = par_max_long((long long)solver_stats->wr_output);
    long long unaccounted_ns =
        par_max_long(local_solve_steps - local_timed_stages);
    long long max_local_cells = par_max_long(local_cells);
    long long comm_ns = par_max_long(local_comm_ns);

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
        (double)eta_ns * ns_to_ms / (double)sample_count;
    double zeta_avg_ms =
        (double)zeta_ns * ns_to_ms / (double)sample_count;
    double u_avg_ms =
        (double)u_ns * ns_to_ms / (double)sample_count;
    double psi_avg_ms =
        (double)psi_ns * ns_to_ms / (double)sample_count;
    double phi_low_avg_ms =
        (double)phi_low_ns * ns_to_ms /
        (double)sample_count;
    double phi_high_avg_ms =
        (double)phi_high_ns * ns_to_ms /
        (double)sample_count;
    double pressure_update_avg_ms =
        (double)pressure_update_ns * ns_to_ms /
        (double)sample_count;
    double wr_output_avg_ms =
        (double)wr_output_ns * ns_to_ms /
        (double)sample_count;
    double porosity_fill_avg_ms =
        (double)porosity_fill_ns * ns_to_ms /
        (double)sample_count;
    double solve_steps_ns = (double)slowest_ns;
    double percentage_factor =
        solve_steps_ns > 0.0 ? 100.0 / solve_steps_ns : 0.0;
    /*
     * Il totale del rank piu' lento meno la sua somma di stadi. Gli stadi
     * qui sotto sono invece massimi presi separatamente: ottimi per trovare
     * un collo di bottiglia, non per essere sommati fra loro.
     */
    double unaccounted_avg_ms =
        unaccounted_ns * ns_to_ms / (double)sample_count;
    double solve_steps_avg_ns = (double)slowest_ns / (double)sample_count;
    double per_cell_step =
        max_local_cells > 0
            ? (solve_steps_avg_ns / (double)max_local_cells) / 10.0
            : 0.0;
    int dims[3];
    par_dims(dims);
    printf("Global grid: %d x %d x %d\n",
           d->n_global[0], d->n_global[1], d->n_global[2]);
    printf("Local block: %d x %d x %d\n", d->n[0], d->n[1], d->n[2]);
    printf("Processes: %d\n", par_size());
    printf("Process grid: %d x %d x %d\n", dims[0], dims[1], dims[2]);
    printf("Critical rank: %d\n", slowest_rank);
    /* Ranghi e thread insieme: una misura senza entrambi i numeri non si sa
     * confrontare con nessun'altra. */
    printf("Threads per process: %d\n", workers_available());
    printf("Tridiagonal backend: %s\n", backend_name());
    if (report_pipeline_batch_lines() > 0) {
        printf("Pipeline batch lines: %d\n", report_pipeline_batch_lines());
    }
    printf("Directional policy: %s\n", workers_line_policy_name());
    printf("Time steps: %zu\n", sample_count);
    printf("Solver time stats (max per rank, average per time step):\n");
    printf("  eta system:  %.3f ms (%5.1f%%)\n", eta_avg_ms,
           (double)eta_ns * percentage_factor);
    /* Le due parti di eta: il termine noto e cio' che resta, cioe' il
     * solutore. Rientrate di due spazi perche' non sono stadi a se': sommarle
     * agli altri stadi conterebbe il passo eta due volte. */
    printf("    g term:    %.3f ms (%5.1f%%)\n",
           (double)g_term_ns * ns_to_ms / (double)sample_count,
           (double)g_term_ns * percentage_factor);
    printf("    eta solve: %.3f ms (%5.1f%%)\n",
           (double)(eta_ns - g_term_ns) * ns_to_ms / (double)sample_count,
           (double)(eta_ns - g_term_ns) * percentage_factor);
    printf("  zeta system: %.3f ms (%5.1f%%)\n", zeta_avg_ms,
           (double)zeta_ns * percentage_factor);
    printf("  u system:    %.3f ms (%5.1f%%)\n", u_avg_ms,
           (double)u_ns * percentage_factor);
    printf("  psi system:  %.3f ms (%5.1f%%)\n", psi_avg_ms,
           (double)psi_ns * percentage_factor);
    printf("  phi low:     %.3f ms (%5.1f%%)\n", phi_low_avg_ms,
           (double)phi_low_ns * percentage_factor);
    printf("  phi high:    %.3f ms (%5.1f%%)\n", phi_high_avg_ms,
           (double)phi_high_ns * percentage_factor);
    printf("  pressure:    %.3f ms (%5.1f%%)\n", pressure_update_avg_ms,
           (double)pressure_update_ns * percentage_factor);
    printf("  porosity:    %.3f ms (%5.1f%%)\n", porosity_fill_avg_ms,
           (double)porosity_fill_ns * percentage_factor);
    printf("  write file:  %.3f ms\n", wr_output_avg_ms);
    /*
     * Quanto del passo non e' attribuito a nessuno stadio.
     *
     * E' una riga di controllo, non una misura: finche' non e' vicina a zero,
     * qualunque conclusione su quale stadio sia lento riguarda solo la parte
     * cronometrata. Il riempimento della permeabilita' e' rimasto fuori dai
     * conti per tutto il tempo, e da solo valeva fino a due terzi del passo.
     */
    printf("  non contato: %.3f ms (%5.1f%%)\n",
           unaccounted_avg_ms, unaccounted_ns * percentage_factor);
    /* Righe pensate per essere lette anche da uno script. */
    printf("  wall per step: %.3f ms\n",
           (double)slowest_ns * ns_to_ms / (double)sample_count);
    printf("  mpi per step:  %.3f ms (%5.1f%%)\n",
           (double)comm_ns * ns_to_ms / (double)sample_count,
           slowest_ns > 0 ? 100.0 * (double)comm_ns / (double)slowest_ns : 0.0);
    printf("  per cell-step: %.3f 1e-8s\n", per_cell_step);
}
