#include "backend.h"
#include "backend_pipeline.h"
#include "parallel.h"
#include "pressure_common.h"
#include "utils.h"

#include <stdbool.h>

/*
 * La cascata di pressione, risolta con lo stesso Thomas pipelined della
 * quantita' di moto.
 *
 * Una differenza rispetto a quella pero' c'e', ed e' la ragione per cui qui
 * non si usa niente di simile a momentum_row: la matrice della pressione e'
 * la stessa per tutte le linee dell'asse e non cambia mai nel tempo.  E' gia'
 * scritta in backend->matrix_a/b/c da backend_init, e ogni punto legge la sua
 * riga per indice invece di ricostruirsela.  Chiederla cella per cella
 * vorrebbe dire ricalcolare a ogni passo temporale una cosa che si sa
 * dall'avvio.
 *
 * L'altra differenza e' che i tre passi sono in cascata -- psi alimenta phi
 * che alimenta fi -- quindi non si possono tenere in volo insieme come le tre
 * componenti della quantita' di moto: ogni direzione e' una pipeline a se',
 * e il riempimento si paga tre volte per passo temporale.
 */

static void pressure_forward(const Decomp *d, PipelineBackend *backend,
                             int axis, const Real *restrict source) {
    const int batch_lines = backend->batch_lines;
    const int length = d->n[axis];
    const size_t line_count = pipeline_line_count(d, axis);
    const size_t batch_count =
        (line_count + (size_t)batch_lines - 1) / (size_t)batch_lines;
    const int tag = 300 + axis;
    const bool has_lower = par_neighbor(axis, -1) != PAR_NO_NEIGHBOR;
    const bool has_upper = par_neighbor(axis, 1) != PAR_NO_NEIGHBOR;
    const Real *restrict row_a = backend->matrix_a[axis];
    const Real *restrict row_b = backend->matrix_b[axis];
    const Real *restrict row_c = backend->matrix_c[axis];

    for (size_t batch = 0; batch < batch_count; batch++) {
        const size_t first_line = batch * (size_t)batch_lines;
        int active = (int)(line_count - first_line);

        if (active > batch_lines) {
            active = batch_lines;
        }
        if (has_lower) {
            par_recv_real(axis, -1, backend->forward, 2 * active, tag);
        }

        for (int level = 0; level < length; level++) {
            for (int line = 0; line < active; line++) {
                int cell[3];
                Real previous_c;
                Real previous_d;

                if (level == 0) {
                    previous_c = has_lower ? backend->forward[line] : (Real)0;
                    previous_d = has_lower ? backend->forward[active + line]
                                           : (Real)0;
                } else {
                    size_t before = pipeline_scratch(backend, axis, 0, batch,
                                                     level - 1, line, length);
                    previous_c = backend->c_prime[before];
                    previous_d = backend->d_prime[before];
                }

                pipeline_cell(d, axis, first_line + (size_t)line, level, cell);

                size_t here = decomp_index(d, cell[0], cell[1], cell[2]);
                Real inverse_diagonal =
                    (Real)1 / (row_b[level] - row_a[level] * previous_c);
                size_t at = pipeline_scratch(backend, axis, 0, batch, level,
                                             line, length);

                backend->c_prime[at] = row_c[level] * inverse_diagonal;
                backend->d_prime[at] =
                    (source[here] - row_a[level] * previous_d) *
                    inverse_diagonal;
            }
        }

        if (has_upper) {
            for (int line = 0; line < active; line++) {
                size_t last = pipeline_scratch(backend, axis, 0, batch,
                                               length - 1, line, length);

                backend->forward[line] = backend->c_prime[last];
                backend->forward[active + line] = backend->d_prime[last];
            }
            par_send_real(axis, 1, backend->forward, 2 * active, tag);
        }
    }
}

static void pressure_backward(const Decomp *d, PipelineBackend *backend,
                              int axis, Real *restrict target) {
    const int batch_lines = backend->batch_lines;
    const int length = d->n[axis];
    const size_t line_count = pipeline_line_count(d, axis);
    const size_t batch_count =
        (line_count + (size_t)batch_lines - 1) / (size_t)batch_lines;
    const int tag = 400 + axis;
    const bool has_lower = par_neighbor(axis, -1) != PAR_NO_NEIGHBOR;
    const bool has_upper = par_neighbor(axis, 1) != PAR_NO_NEIGHBOR;

    for (size_t remaining = batch_count; remaining > 0; remaining--) {
        const size_t batch = remaining - 1;
        const size_t first_line = batch * (size_t)batch_lines;
        int active = (int)(line_count - first_line);

        if (active > batch_lines) {
            active = batch_lines;
        }
        if (has_upper) {
            par_recv_real(axis, 1, backend->backward, active, tag);
        } else {
            for (int line = 0; line < active; line++) {
                backend->backward[line] = (Real)0;
            }
        }

        for (int level = length - 1; level >= 0; level--) {
            for (int line = 0; line < active; line++) {
                int cell[3];
                size_t at = pipeline_scratch(backend, axis, 0, batch, level,
                                             line, length);
                Real solution = backend->d_prime[at] -
                                backend->c_prime[at] * backend->backward[line];

                pipeline_cell(d, axis, first_line + (size_t)line, level, cell);
                /* La pressione assegna, la quantita' di moto sommava: qui la
                 * soluzione e' il campo, non un incremento. */
                target[decomp_index(d, cell[0], cell[1], cell[2])] = solution;
                backend->backward[line] = solution;
            }
        }

        if (has_lower) {
            par_send_real(axis, -1, backend->backward, active, tag);
        }
    }
}

static void pressure_direction(const Decomp *d, PipelineBackend *backend,
                               int axis, const Real *restrict source,
                               Real *restrict target) {
    pressure_forward(d, backend, axis, source);
    pressure_backward(d, backend, axis, target);
}

void pressure_step(const Decomp *d, SolverMemState *solver_mem_state,
                   ScalarField *pressure_buffer, SolverStats *solver_stats) {
    PipelineBackend *backend = solver_mem_state->backend;
    Real *buffer = pressure_buffer->v;
    Real *star = solver_mem_state->pressure_star.v;

    /* I due campi si scambiano il ruolo a ogni passo, cosi' ne bastano due. */
    uint64_t start_ns = time_ns();
    compute_div(d, buffer, &solver_mem_state->u);
    pressure_direction(d, backend, 0, buffer, star);   /* psi */
    solver_stats->psi_sys += time_ns() - start_ns;

    start_ns = time_ns();
    pressure_direction(d, backend, 1, star, buffer);   /* phi */
    solver_stats->phi_low_sys += time_ns() - start_ns;

    start_ns = time_ns();
    pressure_direction(d, backend, 2, buffer, star);   /* fi  */
    solver_stats->phi_high_sys += time_ns() - start_ns;

    start_ns = time_ns();
    update_pressure(d, solver_mem_state);
    solver_stats->pressure_update += time_ns() - start_ns;
}
