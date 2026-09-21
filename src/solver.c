#include "solver.h"
#include "field.h"
#include "backend.h"
#include "output.h"
#include "parallel.h"
#include "workers.h"

/*
 * Refreshes the ring of boundary cells of the fields that the computations
 * read one step beyond their own block.
 *
 * g_value computes three second derivatives, one per axis: it looks at eta
 * along X, zeta along Y and u along Z, plus the extrapolated pressure in all
 * three. The other fields are read only in the cell where they sit.
 */
static void refresh_vector_halo(const Decomp *d, VectorField *field) {
    par_exchange_halo(d, field->v_x);
    par_exchange_halo(d, field->v_y);
    par_exchange_halo(d, field->v_z);
}

/*
 * Whoever has a neighbour above also writes the first plane of cells of the
 * neighbour, which it reads from the boundary ring: without that shared plane
 * the pieces of the .pvti file would leave a row of cells uncovered at every
 * interface.
 *
 * The ring must therefore be refreshed right before writing: the pressure has
 * just been rewritten without an exchange, the permeability never makes one,
 * and at step zero none has happened yet.
 *
 * It is also needed for carrying the velocity over to the cell-centred nodes,
 * which on the boundary between two blocks averages with the cell just beyond.
 */
static void refresh_before_write(const Decomp *d, SolverMemState *state) {
    par_exchange_halo(d, state->pressure.v);
    refresh_vector_halo(d, &state->u);
    refresh_vector_halo(d, &state->k);
}

void solver_init(const Decomp *decomp,
                 SolverMemState *solver_mem_state,
                 Data *data,
                 const char *data_name) {
    /* Without a name the scenario that the caller has already put in `data` is
     * kept: that is what the tests do, since they build their own. */
    if (data_name != NULL) {
        const Data *found = data_by_name(data_name);

        if (found == NULL) {
            if (par_rank() == 0) {
                fprintf(stderr, "Unknown scenario: %s\nAvailable:\n",
                        data_name);
                data_print_names(stderr);
            }
            par_abort(1);
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

    /* The scratch needed by the tridiagonal backend is allocated by the
     * backend itself: from here it is an opaque pointer, and which of the two
     * it is cannot be seen. */
    backend_init(decomp, solver_mem_state);

    /* Writing carries the velocity from the staggered points to the nodes, and
     * on the boundary between two blocks the average asks for the cell just
     * beyond: the boundary ring must be refreshed before every dump, not only
     * before the computations. */
    if (write_enabled) {
        refresh_before_write(decomp, solver_mem_state);
        write_to_file(decomp, solver_mem_state, data->name, 0);
    }

    uint64_t comm_start = par_comm_nanoseconds();
    uint64_t output_comm = 0;
    uint64_t start_ns = time_ns();
    // N.B. the loop starts from t=1, because step t=0 has already been written
    // above
    for (int t_step = 1; t_step <= STEPS; t_step++) {
        // the boundary ring is refreshed again for the 4 values needed by g
        refresh_vector_halo(decomp, &solver_mem_state->eta);
        refresh_vector_halo(decomp, &solver_mem_state->zeta);
        refresh_vector_halo(decomp, &solver_mem_state->u);
        par_exchange_halo(decomp, solver_mem_state->pressure_star.v);

        // if the porosity depends on time, it is updated at the middle of the
        // time step
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
        // updates the velocity and pressure_star, using the current pressure
        // and porosity values
        momentum_step(decomp, solver_mem_state, data, t_step, solver_stats);


        /* compute_div looks one cell back in the velocity just updated, on all
         * three axes. Otherwise the divergence on the lower edge of the block
         * would be one step old.
         */
        // updates the boundary ring of the velocity just computed: compute_div
        // reads it, in the pressure_step below
        refresh_vector_halo(decomp, &solver_mem_state->u);

        // Pressure system
        // updates the pressure, using the current velocity and porosity values
        pressure_step(decomp, solver_mem_state, &pressure_buffer,
                      solver_stats);

        // Write to file
        // writes to file the current values of velocity, pressure and porosity,
        // if write_enabled is set
        if (write_enabled) {
            if (t_step % WR_FREQ == 0) {
                uint64_t wr_start = time_ns();
                uint64_t wr_comm = par_comm_nanoseconds();
                /* pressure_step has just corrected u: the boundary ring still
                 * carries the values from before the projection. */
                refresh_before_write(decomp, solver_mem_state);
                write_to_file(decomp, solver_mem_state, data->name, t_step);
                solver_stats->wr_output += time_ns() - wr_start;
                output_comm += par_comm_nanoseconds() - wr_comm;
            }
        }
    }

    // updates the total execution time of the solver, subtracting the time
    // spent writing to file
    solver_stats->solve_steps = (time_ns() - start_ns) - solver_stats->wr_output;

    solver_stats->comm_steps = par_comm_nanoseconds() - comm_start - output_comm;

    // Print solver time statistics
    print_stats(decomp, solver_stats, (size_t)STEPS);

    backend_free(solver_mem_state);
    free(pressure_buffer.v);
}
