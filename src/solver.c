#include "solver.h"
#include "field.h"
#include "backend.h"
#include "output.h"
#include "parallel.h"
#include "workers.h"

/*
 * Rinfresca l'anello di celle di contorno dei campi che i conti leggono un
 * passo oltre il proprio blocco.
 *
 * g_value calcola tre derivate seconde, una per asse: guarda eta lungo X,
 * zeta lungo Y e u lungo Z, piu' la pressione estrapolata in tutte e tre.
 * Gli altri campi vengono letti solo nella cella in cui si trovano.
 */
static void refresh_vector_halo(const Decomp *d, VectorField *field) {
    par_exchange_halo(d, field->v_x);
    par_exchange_halo(d, field->v_y);
    par_exchange_halo(d, field->v_z);
}

void solver_init(const Decomp *decomp,
                 SolverMemState *solver_mem_state,
                 Data *data,
                 const char *data_name) {
    /* Senza nome si tiene lo scenario che il chiamante ha gia' messo in `data`:
     * e' quello che fanno i test, che costruiscono il proprio. */
    if (data_name != NULL) {
        const Data *found = data_by_name(data_name);

        if (found == NULL) {
            if (par_rank() == 0) {
                fprintf(stderr, "Scenario sconosciuto: %s\nDisponibili:\n",
                        data_name);
                data_print_names(stderr);
            }
            exit(1);
        }

        *data = *found;
    }

    // Allocate memory
    scalarField_alloc(decomp, &solver_mem_state->pressure);
    scalarField_alloc(decomp, &solver_mem_state->pressure_star);
    vectorField_alloc(decomp, &solver_mem_state->k);
    vectorField_alloc(decomp, &solver_mem_state->eta);
    vectorField_alloc(decomp, &solver_mem_state->zeta);
    vectorField_alloc(decomp, &solver_mem_state->u);

    // Fill velocity with values at t=0
    vectorField_fill(decomp, &solver_mem_state->eta, data->velocity_fn, 0);
    vectorField_fill(decomp, &solver_mem_state->zeta, data->velocity_fn, 0);
    vectorField_fill(decomp, &solver_mem_state->u, data->velocity_fn, 0);

    // Fill pressure with values at t=0
    scalarField_fill(decomp, &solver_mem_state->pressure,
                     data->pressure_fn, 0);
    scalarField_fill(decomp, &solver_mem_state->pressure_star,
                     data->pressure_fn, 0);

    // Fill the initial porosity field at t=0
    vectorField_fill(decomp, &solver_mem_state->k, data->porosity_fn, 0);
}

void solver_solve(const Decomp *decomp, SolverMemState *solver_mem_state,
                  Data *data, SolverStats *solver_stats,
                  int write_enabled) {
    // Used for the pressure_step, this buffer and pressure_star are sufficient to solve it
    ScalarField pressure_buffer;
    scalarField_alloc(decomp, &pressure_buffer);

    /* Lo scratch che serve al backend tridiagonale lo alloca lui: da qui e'
     * un puntatore opaco, e quale dei due sia non si vede. */
    backend_init(decomp, solver_mem_state);

    /* La scrittura riporta la velocita' dai punti staggered ai nodi, e sul
     * confine fra due blocchi la media chiede la cella appena oltre: l'anello
     * di contorno va aggiornato prima di ogni dump, non solo prima dei conti. */
    if (write_enabled) {
        refresh_vector_halo(decomp, &solver_mem_state->u);
        write_to_file(decomp, solver_mem_state, data->name, 0);
    }

    uint64_t start_ns = time_ns();
    for (int t_step = 1; t_step <= STEPS; t_step++) {
        refresh_vector_halo(decomp, &solver_mem_state->eta);
        refresh_vector_halo(decomp, &solver_mem_state->zeta);
        refresh_vector_halo(decomp, &solver_mem_state->u);
        par_exchange_halo(decomp, solver_mem_state->pressure_star.v);

        if (data->porosity_time_dependent) {
            uint64_t fill_start = time_ns();
            Real midpoint_step = (Real)t_step - (Real)0.5;
            vectorField_fill(decomp,
                             &solver_mem_state->k,
                             data->porosity_fn,
                             midpoint_step);
            solver_stats->porosity_fill += time_ns() - fill_start;
        }

        // Momentum system
        momentum_step(decomp, solver_mem_state, data, t_step, solver_stats);


        /* compute_div guarda una cella indietro nella velocita' appena
         * aggiornata, su tutti e tre gli assi. */
        refresh_vector_halo(decomp, &solver_mem_state->u);

        // Pressure system
        pressure_step(decomp, solver_mem_state, &pressure_buffer,
                      solver_stats);

        // Write to file
        if (write_enabled) {
            if (t_step % WR_FREQ == 0) {
                uint64_t wr_start = time_ns();
                /* pressure_step ha appena corretto u: l'anello di contorno
                 * porta ancora i valori di prima della proiezione. */
                refresh_vector_halo(decomp, &solver_mem_state->u);
                write_to_file(decomp, solver_mem_state, data->name, t_step);
                solver_stats->wr_output += time_ns() - wr_start;
            }
        }
    }
    solver_stats->solve_steps = (time_ns() - start_ns) - solver_stats->wr_output;

    // Print solver time statistics
    print_stats(decomp, solver_stats, (size_t)STEPS);

    backend_free(solver_mem_state);
    free(pressure_buffer.v);
}
