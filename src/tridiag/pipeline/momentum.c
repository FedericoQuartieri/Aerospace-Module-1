#include "backend.h"
#include "backend_pipeline.h"
#include "momentum_row.h"
#include "parallel.h"
#include "simd_real.h"
#include "utils.h"
#include "workers.h"

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
 *
 * Con SIMD=1 le passate lungo Y e Z vettorizzano *attraverso* le linee: in
 * quelle direzioni le linee consecutive sono adiacenti nello scratch e nel
 * campo (pipeline_scratch, pipeline_cell), quindi un vettore tiene lo stesso
 * livello di SIMD_LANES linee vicine.  E' il kernel che aveva il branch
 * pipeline, riportato pari pari: copre i punti interni di Y e Z, e lascia al
 * percorso scalare i due punti di parete, le linee avanzate in fondo a un
 * batch o a una riga, e tutta la direzione X, dove le linee non sono
 * adiacenti.  I due percorsi fanno le stesse operazioni nello stesso ordine,
 * per cui il risultato non cambia di un bit.
 */

static Real *target_field(SolverMemState *state, int axis, int component) {
    VectorField *to = (axis == 0) ? &state->eta
                    : (axis == 1) ? &state->zeta
                                  : &state->u;
    return (component == 0) ? to->v_x
         : (component == 1) ? to->v_y
                            : to->v_z;
}

#if defined(USE_SIMD) && SIMD_AVAILABLE

/*
 * Il coefficiente w di Thomas da un vettore di permeabilita': le stesse
 * operazioni, nello stesso ordine, di gamma_from_k, ed e' questo che tiene
 * il risultato identico a quello scalare.
 */
static SimdReal momentum_weight_simd(SimdReal permeability,
                                     Real inverse_square) {
    SimdReal one = simd_set1((Real)1);
    SimdReal two = simd_set1((Real)2);
    SimdReal numerator = simd_set1((Real)(DT * NU));
    SimdReal beta = simd_add(
        one, simd_div(numerator, simd_mul(two, permeability)));
    SimdReal gamma = simd_div(numerator, simd_mul(two, beta));

    return simd_mul(gamma, simd_set1(-inverse_square));
}

/*
 * Eliminazione in avanti di SIMD_LANES linee consecutive allo stesso
 * livello, a partire da `line`.  Vale solo per un punto interno lungo Y o Z:
 * li' la riga e' (w, 1-2w, w, source-target), senza termine g e senza
 * condizioni al contorno, ed e' l'unica che si scrive in quattro operazioni.
 * Restituisce quante linee ha fatto: SIMD_LANES, oppure 0 quando non puo' --
 * punto di parete, meno di un vettore di linee ancora attive, o linee che
 * scavalcherebbero la fine di una riga lungo X e non sarebbero piu' adiacenti
 * nel campo.  In quel caso ci pensa il percorso scalare, linea per linea.
 *
 * Al livello 0 legge la giunzione ricevuta dal vicino di sotto senza
 * chiedersi se esista: se non esiste, il livello 0 e' la parete inferiore
 * globale e questa funzione ha gia' risposto 0.
 */
static int forward_interior_simd(const Decomp *d, PipelineBackend *backend,
                                 const MomentumLine *line_ctx, int axis,
                                 int component, size_t batch,
                                 size_t first_line, int active, int level,
                                 int line, int length) {
    int cell[3];
    int global_axis = decomp_global(d, level, axis);

    if (global_axis == 0 || global_axis == d->n_global[axis] - 1 ||
        line + SIMD_LANES > active) {
        return 0;
    }

    pipeline_cell(d, axis, first_line + (size_t)line, level, cell);
    if (cell[0] + SIMD_LANES > d->n[0]) {
        return 0;
    }

    size_t here = decomp_index(d, cell[0], cell[1], cell[2]);
    size_t at = pipeline_scratch(backend, axis, component, batch, level,
                                 line, length);
    const Real *previous_c;
    const Real *previous_d;

    if (level == 0) {
        previous_c = &backend->forward[line];
        previous_d = &backend->forward[active + line];
    } else {
        size_t before = pipeline_scratch(backend, axis, component, batch,
                                         level - 1, line, length);
        previous_c = &backend->c_prime[before];
        previous_d = &backend->d_prime[before];
    }

    SimdReal one = simd_set1((Real)1);
    SimdReal two = simd_set1((Real)2);
    SimdReal w = momentum_weight_simd(simd_loadu(&line_ctx->k_porosity[here]),
                                      line_ctx->inverse_square);
    SimdReal inverse_diagonal = simd_div(
        one, simd_sub(simd_sub(one, simd_mul(two, w)),
                      simd_mul(w, simd_loadu(previous_c))));
    SimdReal raw_rhs = simd_sub(simd_loadu(&line_ctx->source[here]),
                                simd_loadu(&line_ctx->target[here]));

    simd_storeu(&backend->c_prime[at], simd_mul(w, inverse_diagonal));
    simd_storeu(&backend->d_prime[at],
                simd_mul(simd_sub(raw_rhs,
                                  simd_mul(w, simd_loadu(previous_d))),
                         inverse_diagonal));
    return SIMD_LANES;
}

/*
 * Sostituzione all'indietro di SIMD_LANES linee consecutive allo stesso
 * livello.  Qui non c'e' fisica e non c'e' parete: le condizioni sono solo
 * quelle di adiacenza, e sono le stesse dell'andata.
 */
static int backward_simd(const Decomp *d, PipelineBackend *backend,
                         Real *restrict target, int axis, int component,
                         size_t batch, size_t first_line, int active,
                         int level, int line, int length) {
    int cell[3];

    if (line + SIMD_LANES > active) {
        return 0;
    }

    pipeline_cell(d, axis, first_line + (size_t)line, level, cell);
    if (cell[0] + SIMD_LANES > d->n[0]) {
        return 0;
    }

    size_t here = decomp_index(d, cell[0], cell[1], cell[2]);
    size_t at = pipeline_scratch(backend, axis, component, batch, level,
                                 line, length);
    SimdReal solution = simd_sub(
        simd_loadu(&backend->d_prime[at]),
        simd_mul(simd_loadu(&backend->c_prime[at]),
                 simd_loadu(&backend->backward[line])));

    simd_storeu(&target[here],
                simd_add(simd_loadu(&target[here]), solution));
    simd_storeu(&backend->backward[line], solution);
    return SIMD_LANES;
}

#endif

/*
 * Le linee si percorrono a gruppi larghi un vettore: il kernel SIMD prende
 * un gruppo intero, e quando non puo' ogni linea del gruppo passa dal
 * percorso scalare.  Lungo X il gruppo e' una linea sola, perche' li' le
 * linee non sono adiacenti ne' nello scratch ne' nel campo.  Senza SIMD il
 * gruppo e' sempre una linea e il ciclo e' quello di prima.  Il passo e'
 * costante dentro il ciclo, che e' quello che serve a WORKERS_FOR per
 * spartirlo fra i thread.
 */
static int line_step_for(int axis) {
#if defined(USE_SIMD) && SIMD_AVAILABLE
    return (axis == 0) ? 1 : SIMD_LANES;
#else
    (void)axis;
    return 1;
#endif
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
                              int t_step, SolverStats *solver_stats) {
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
    const int line_step = line_step_for(axis);

    /*
     * Le ascisse non dipendono ne' dalla linea ne' dal livello ne' dal passo,
     * solo dalla componente: un riempimento serve tutti i batch.
     */
    if (axis == 0) {
        forcing_line_coords(d, component, backend->abscissa);
    }

    for (size_t batch = 0; batch < batch_count; batch++) {
        const size_t first_line = batch * (size_t)batch_lines;
        int active = (int)(line_count - first_line);

        if (active > batch_lines) {
            active = batch_lines;
        }
        if (has_lower) {
            par_recv_real(axis, -1, backend->forward, 2 * active, tag);
        }

        /*
         * Il termine fisico g delle linee del batch, prima di scendere nei
         * livelli.  Solo lo stadio eta lo porta, e le sue linee corrono lungo
         * x: su una linea y, z e il tempo sono costanti, quindi i test che
         * governano g si decidono una volta e il resto si vettorizza.  Prima
         * si arrivava qui con source_term a NULL e momentum_row rivalutava la
         * forzante cella per cella (momentum_row.h).
         *
         * Cronometrato come nel backend Schur: due letture dell'orologio per
         * batch, e il tempo e' gia' di parete perche' stanno fuori dal team.
         */
        if (axis == 0) {
            uint64_t inizio = time_ns();

            WORKERS_PARALLEL_FOR(workers_many() && active > 1)
            for (int line = 0; line < active; line++) {
                int cell[3];

                pipeline_cell(d, axis, first_line + (size_t)line, 0, cell);
                g_line(d, data, state, line_ctx.k_porosity,
                       cell[1], cell[2], t_step, component,
                       backend->abscissa,
                       backend->source_term + (size_t)line * (size_t)length);
            }

            solver_stats->momentum_source += time_ns() - inizio;
        }

        /*
         * Dentro un batch le linee non si parlano: l'unica dipendenza e'
         * lungo `level` nella stessa linea.  Per questo i thread spartiscono
         * le linee a ogni livello, mentre la recv e la send restano fuori dal
         * team e mantengono identico l'ordine dei messaggi MPI.
         */
        WORKERS_PARALLEL(workers_many() && active > 1)
        {
            for (int level = 0; level < length; level++) {
                WORKERS_FOR
                for (int first = 0; first < active; first += line_step) {
                    int last = first + line_step;

                    if (last > active) {
                        last = active;
                    }
#if defined(USE_SIMD) && SIMD_AVAILABLE
                    if (line_step > 1 &&
                        forward_interior_simd(d, backend, &line_ctx, axis,
                                              component, batch, first_line,
                                              active, level, first,
                                              length) > 0) {
                        continue;
                    }
#endif
                    for (int line = first; line < last; line++) {
                        int cell[3];
                        Real previous_c;
                        Real previous_d;

                        if (level == 0) {
                            previous_c =
                                has_lower ? backend->forward[line] : (Real)0;
                            previous_d =
                                has_lower ? backend->forward[active + line]
                                          : (Real)0;
                        } else {
                            size_t before =
                                pipeline_scratch(backend, axis, component,
                                                 batch, level - 1, line,
                                                 length);
                            previous_c = backend->c_prime[before];
                            previous_d = backend->d_prime[before];
                        }

                        pipeline_cell(d, axis, first_line + (size_t)line,
                                      level, cell);

                        size_t here =
                            decomp_index(d, cell[0], cell[1], cell[2]);
                        /*
                         * Lungo x il g della linea e' gia' nel buffer del
                         * batch: si punta la sua riga e momentum_row lo legge
                         * invece di rivalutare la forzante.  Sugli altri due
                         * assi g non entra e il contesto resta quello.
                         */
                        MomentumLine cell_ctx = line_ctx;

                        if (axis == 0) {
                            cell_ctx.source_term =
                                backend->source_term +
                                (size_t)line * (size_t)length;
                        }

                        MomentumRow row = momentum_row(&cell_ctx, cell, here);
                        Real inverse_diagonal =
                            (Real)1 / (row.b - row.a * previous_c);
                        size_t at = pipeline_scratch(backend, axis, component,
                                                     batch, level, line,
                                                     length);

                        backend->c_prime[at] = row.c * inverse_diagonal;
                        backend->d_prime[at] =
                            (row.f - row.a * previous_d) * inverse_diagonal;
                    }
                }
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
    const int line_step = line_step_for(axis);

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

        WORKERS_PARALLEL(workers_many() && active > 1)
        {
            for (int level = length - 1; level >= 0; level--) {
                WORKERS_FOR
                for (int first = 0; first < active; first += line_step) {
                    int last = first + line_step;

                    if (last > active) {
                        last = active;
                    }
#if defined(USE_SIMD) && SIMD_AVAILABLE
                    if (line_step > 1 &&
                        backward_simd(d, backend, target, axis, component,
                                      batch, first_line, active, level,
                                      first, length) > 0) {
                        continue;
                    }
#endif
                    for (int line = first; line < last; line++) {
                        int cell[3];
                        size_t at = pipeline_scratch(backend, axis, component,
                                                     batch, level, line,
                                                     length);
                        Real solution =
                            backend->d_prime[at] -
                            backend->c_prime[at] * backend->backward[line];

                        pipeline_cell(d, axis, first_line + (size_t)line,
                                      level, cell);
                        target[decomp_index(d, cell[0], cell[1], cell[2])] +=
                            solution;
                        backend->backward[line] = solution;
                    }
                }
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
                               Data *data, int axis, int t_step,
                               SolverStats *solver_stats) {
    for (int component = 0; component < 3; component++) {
        forward_component(d, state, data, axis, component, t_step,
                          solver_stats);
    }
    for (int component = 2; component >= 0; component--) {
        backward_component(d, state, axis, component);
    }
}

void momentum_step(const Decomp *d, SolverMemState *solver_mem_state,
                   Data *data, int t_step, SolverStats *solver_stats) {
    uint64_t start_ns = time_ns();

    momentum_direction(d, solver_mem_state, data, 0, t_step,
                       solver_stats);
    solver_stats->eta_sys += time_ns() - start_ns;

    start_ns = time_ns();
    momentum_direction(d, solver_mem_state, data, 1, t_step,
                       solver_stats);
    solver_stats->zeta_sys += time_ns() - start_ns;

    start_ns = time_ns();
    momentum_direction(d, solver_mem_state, data, 2, t_step,
                       solver_stats);
    solver_stats->u_sys += time_ns() - start_ns;
}
