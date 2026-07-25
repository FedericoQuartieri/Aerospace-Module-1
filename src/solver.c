#include "solver.h"
#include "field.h"
#include "momentum.h"
#include "pressure.h"
#include "output.h"

void solver_init(SolverMemState *solver_mem_state,
                 Data *data,
                 const char *data_name) {
    (void)data_name;
   
    // TODO: Parse data_name and assign the correspondent Data structure,
    // return error if not matched
    
    // Allocate memory
    scalarField_alloc(&solver_mem_state->pressure);
    scalarField_alloc(&solver_mem_state->pressure_star);
    vectorField_alloc(&solver_mem_state->k);
    vectorField_alloc(&solver_mem_state->eta);
    vectorField_alloc(&solver_mem_state->zeta);
    vectorField_alloc(&solver_mem_state->u);
    
    // Fill velocity with values at t=0
    vectorField_fill(&solver_mem_state->eta, data->velocity_fn, 0);
    vectorField_fill(&solver_mem_state->zeta, data->velocity_fn, 0);
    vectorField_fill(&solver_mem_state->u, data->velocity_fn, 0);
    
    // Fill pressure with values at t=0
    scalarField_fill(&solver_mem_state->pressure, data->pressure_fn, 0);
    scalarField_fill(&solver_mem_state->pressure_star, data->pressure_fn, 0);
    
    // Fill porosity field (currently is supposed to be constant in time)
    vectorField_fill(&solver_mem_state->k, data->porosity_fn, 0);
}

void solver_solve(SolverMemState *solver_mem_state, Data *data, SolverStats *solver_stats,
                  int write_enabled) {
    // This buffer allocate the biggest grid dimension, in order
    // to be shared among the component.
    size_t big_dim = (WIDTH > HEIGHT) ? WIDTH : HEIGHT;
    big_dim = (big_dim > DEPTH) ? big_dim : DEPTH;
    Real *restrict rhs = xmalloc(big_dim * sizeof(Real));
    Real *restrict tmp = xmalloc(big_dim * sizeof(Real));

    // Used for the pressure_step, this buffer and pressure_star are sufficient to solve it
    ScalarField pressure_buffer;
    scalarField_alloc(&pressure_buffer);

    // If write_enabled is set, write the initial state at t=0
    if (write_enabled) write_to_file(solver_mem_state, 0);

    uint64_t start_ns = time_ns();
    for (int t_step = 1; t_step <= STEPS; t_step++) {
        // Momentum system
        momentum_step(solver_mem_state, rhs, tmp, data, t_step, solver_stats);


        // Pressure system
        pressure_step(solver_mem_state, &pressure_buffer, rhs, tmp, solver_stats);

        // Write to file
        if (write_enabled) {
            if (t_step % WR_FREQ == 0) {
                uint64_t wr_start = time_ns();
                write_to_file(solver_mem_state, t_step);
                solver_stats->wr_output += time_ns() - wr_start;
            }
        }
    }
    solver_stats->solve_steps = (time_ns() - start_ns) - solver_stats->wr_output;

    // Print solver time statistics
    print_stats(solver_stats, (size_t)STEPS);

    free(rhs);
    free(tmp);
    free(pressure_buffer.v);
}
