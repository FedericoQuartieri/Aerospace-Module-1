#include "solver.h"
#include "field.h"
#include "momentum.h"
#include "pressure.h"
#include "output.h"
#include "parallel.h"

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
    (void)data_name;

    // TODO: Parse data_name and assign the correspondent Data structure,
    // return error if not matched

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
    // The scalar solvers use one grid line. SIMD momentum stores several
    // independent lines interleaved in the same reusable buffers.
    size_t big_dim = (size_t)((decomp->n[0] > decomp->n[1])
                                  ? decomp->n[0] : decomp->n[1]);
    big_dim = (big_dim > (size_t)decomp->n[2]) ? big_dim
                                               : (size_t)decomp->n[2];
    size_t scratch_size = big_dim * MOMENTUM_SIMD_MAX_LINES;
    Real *restrict rhs = xmalloc(scratch_size * sizeof(Real));
    Real *restrict tmp = xmalloc(scratch_size * sizeof(Real));

    // Used for the pressure_step, this buffer and pressure_star are sufficient to solve it
    ScalarField pressure_buffer;
    scalarField_alloc(decomp, &pressure_buffer);

    // If write_enabled is set, write the initial state at t=0
    if (write_enabled) {
        write_to_file(decomp, solver_mem_state, data->name, 0);
    }

    uint64_t start_ns = time_ns();
    for (int t_step = 1; t_step <= STEPS; t_step++) {
        refresh_vector_halo(decomp, &solver_mem_state->eta);
        refresh_vector_halo(decomp, &solver_mem_state->zeta);
        refresh_vector_halo(decomp, &solver_mem_state->u);
        par_exchange_halo(decomp, solver_mem_state->pressure_star.v);

        if (data->porosity_time_dependent) {
            Real midpoint_step = (Real)t_step - (Real)0.5;
            vectorField_fill(decomp,
                             &solver_mem_state->k,
                             data->porosity_fn,
                             midpoint_step);
        }

        // Momentum system
        momentum_step(decomp, solver_mem_state, rhs, tmp, data, t_step,
                      solver_stats);


        /* compute_div guarda una cella indietro nella velocita' appena
         * aggiornata, su tutti e tre gli assi. */
        refresh_vector_halo(decomp, &solver_mem_state->u);

        // Pressure system
        pressure_step(decomp, solver_mem_state, &pressure_buffer, rhs, tmp,
                      solver_stats);

        // Write to file
        if (write_enabled) {
            if (t_step % WR_FREQ == 0) {
                uint64_t wr_start = time_ns();
                write_to_file(decomp, solver_mem_state, data->name, t_step);
                solver_stats->wr_output += time_ns() - wr_start;
            }
        }
    }
    solver_stats->solve_steps = (time_ns() - start_ns) - solver_stats->wr_output;

    // Print solver time statistics
    print_stats(decomp, solver_stats, (size_t)STEPS);

    free(rhs);
    free(tmp);
    free(pressure_buffer.v);
}
