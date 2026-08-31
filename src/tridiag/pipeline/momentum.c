#include "backend.h"
#include "backend_pipeline.h"
#include "momentum_row.h"
#include "parallel.h"
#include "utils.h"

#include <stdbool.h>

/*
 * I tre sistemi della quantita' di moto, risolti con Thomas pipelined.
 *
 * La differenza con il complemento di Schur non e' nella fisica -- quella e'
 * la stessa e sta in momentum_row.h -- ma in cosa si fa con una linea che
 * attraversa piu' processi.  Schur la spezza in blocchi e ricuce le giunzioni
 * con un sistema piccolo, pagando tre risoluzioni locali invece di una.  Qui
 * la linea non si spezza affatto: si tiene Thomas com'e' e si nasconde
 * l'attesa mandando in pipeline batch di linee *diverse*, che sono
 * indipendenti fra loro per costruzione.
 *
 * L'eliminazione e' quella di sempre,
 *
 *     inv  = 1 / (b - a * c'[i-1])
 *     c'[i] = c * inv
 *     d'[i] = (f - a * d'[i-1]) * inv
 *
 * e le quattro (a, b, c, f) arrivano da momentum_row inline: nascono e
 * muoiono nei registri, non toccano mai la memoria.  E' il punto per cui la
 * giuntura fra i due backend sta al passo direzionale e non piu' in basso --
 * una firma che chiedesse gli array a, b, c, f obbligherebbe a scriverli.
 */

static Real *target_field(SolverMemState *state, int axis, int component) {
    VectorField *to = (axis == 0) ? &state->eta
                    : (axis == 1) ? &state->zeta
                                  : &state->u;
    return (component == 0) ? to->v_x
         : (component == 1) ? to->v_y
                            : to->v_z;
}

/*
 * Passata avanti su una componente, tutti i batch in fila.
 *
 * Riceve dal vicino di sotto i (c', d') della sua ultima riga, elimina il
 * proprio pezzo di linea, e passa i propri a quello di sopra.  Fra la recv e
 * la send c'e' il calcolo, ed e' quello che tiene occupati gli altri: quando
 * il processo 0 e' al batch 3, il processo 3 e' al batch 0.
 */
static void forward_component(const Decomp *d, SolverMemState *state,
                              Data *data, int axis, int component,
                              int t_step) {
    PipelineBackend *backend = state->backend;
    const int batch_lines = backend->batch_lines;
    const int length = d->n[axis];
    const size_t line_count = pipeline_line_count(d, axis);
    const size_t batch_count =
        (line_count + (size_t)batch_lines - 1) / (size_t)batch_lines;
    const int tag = 100 + 4 * axis + component;
    const bool has_lower = par_neighbor(axis, -1) != PAR_NO_NEIGHBOR;
    const bool has_upper = par_neighbor(axis, 1) != PAR_NO_NEIGHBOR;
    const MomentumLine line_ctx =
        momentum_line(d, state, data, t_step, component, axis);

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
                    size_t before = pipeline_scratch(backend, axis, component,
                                                     batch, level - 1, line,
                                                     length);
                    previous_c = backend->c_prime[before];
                    previous_d = backend->d_prime[before];
                }

                pipeline_cell(d, axis, first_line + (size_t)line, level, cell);

                size_t here = decomp_index(d, cell[0], cell[1], cell[2]);
                MomentumRow row = momentum_row(&line_ctx, cell, here);
                Real inverse_diagonal = (Real)1 / (row.b - row.a * previous_c);
                size_t at = pipeline_scratch(backend, axis, component, batch,
                                             level, line, length);

                backend->c_prime[at] = row.c * inverse_diagonal;
                backend->d_prime[at] =
                    (row.f - row.a * previous_d) * inverse_diagonal;
            }
        }

        if (has_upper) {
            for (int line = 0; line < active; line++) {
                size_t last = pipeline_scratch(backend, axis, component, batch,
                                               length - 1, line, length);

                backend->forward[line] = backend->c_prime[last];
                backend->forward[active + line] = backend->d_prime[last];
            }
            par_send_real(axis, 1, backend->forward, 2 * active, tag);
        }
    }
}

/*
 * Passata indietro, nel verso opposto: riceve da sopra, sostituisce, manda
 * sotto.  I batch si ripercorrono dall'ultimo al primo, cosi' il processo che
 * ha finito per ultimo la passata avanti e' il primo a poter cominciare
 * questa.
 */
static void backward_component(const Decomp *d, SolverMemState *state,
                               int axis, int component) {
    PipelineBackend *backend = state->backend;
    const int batch_lines = backend->batch_lines;
    const int length = d->n[axis];
    const size_t line_count = pipeline_line_count(d, axis);
    const size_t batch_count =
        (line_count + (size_t)batch_lines - 1) / (size_t)batch_lines;
    const int tag = 200 + 4 * axis + component;
    const bool has_lower = par_neighbor(axis, -1) != PAR_NO_NEIGHBOR;
    const bool has_upper = par_neighbor(axis, 1) != PAR_NO_NEIGHBOR;
    Real *restrict target = target_field(state, axis, component);

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
            /* Ultimo blocco della linea: oltre l'ultimo punto non c'e' nulla. */
            for (int line = 0; line < active; line++) {
                backend->backward[line] = (Real)0;
            }
        }

        for (int level = length - 1; level >= 0; level--) {
            for (int line = 0; line < active; line++) {
                int cell[3];
                size_t at = pipeline_scratch(backend, axis, component, batch,
                                             level, line, length);
                Real solution = backend->d_prime[at] -
                                backend->c_prime[at] * backend->backward[line];

                pipeline_cell(d, axis, first_line + (size_t)line, level, cell);
                target[decomp_index(d, cell[0], cell[1], cell[2])] += solution;
                backend->backward[line] = solution;
            }
        }

        if (has_lower) {
            par_send_real(axis, -1, backend->backward, active, tag);
        }
    }
}

/*
 * Avanti x, y, z -- indietro z, y, x.
 *
 * L'ordine inverso non e' un vezzo.  La passata avanti risale la catena dei
 * processi, quella indietro la ridiscende: l'ultima cosa che fa l'avanti e'
 * finire z sull'ultimo processo, e la prima cosa che fa l'indietro e'
 * cominciare z sullo stesso processo.  Cosi' la pipeline si gira su se' stessa
 * invece di svuotarsi e riempirsi di nuovo, e il riempimento si paga due volte
 * per direzione invece di sei.
 */
static void momentum_direction(const Decomp *d, SolverMemState *state,
                               Data *data, int axis, int t_step) {
    for (int component = 0; component < 3; component++) {
        forward_component(d, state, data, axis, component, t_step);
    }
    for (int component = 2; component >= 0; component--) {
        backward_component(d, state, axis, component);
    }
}

void momentum_step(const Decomp *d, SolverMemState *solver_mem_state,
                   Data *data, int t_step, SolverStats *solver_stats) {
    uint64_t start_ns = time_ns();

    momentum_direction(d, solver_mem_state, data, 0, t_step);
    solver_stats->eta_sys += time_ns() - start_ns;

    start_ns = time_ns();
    momentum_direction(d, solver_mem_state, data, 1, t_step);
    solver_stats->zeta_sys += time_ns() - start_ns;

    start_ns = time_ns();
    momentum_direction(d, solver_mem_state, data, 2, t_step);
    solver_stats->u_sys += time_ns() - start_ns;
}
