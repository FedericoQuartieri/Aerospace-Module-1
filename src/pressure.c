/* DESIGN: pressure alternates between one workspace field and pressure_star.
 * The latter is dead after momentum and is restored as the next predictor. */
#include "kernels.h"
#include "solver_internal.h"

void pressure_finish_step(const Grid *grid,
                          ScalarField *pressure,
                          ScalarField *pressure_pipeline)
{
    size_t q;
    for (q = 0; q < grid->cell_count; ++q) {
        const Real correction = pressure_pipeline->data[q];
        const Real pressure_next = pressure->data[q] + correction;
        pressure->data[q] = pressure_next;
        /* The baseline predictor is p*_(n+1/2) = p_(n+1/2) + phi. */
        pressure_pipeline->data[q] = pressure_next + correction;
    }
}

void pressure_step(const Grid *grid,
                   const SolverConfig *config,
                   SolverState *state,
                   SolverWorkspace *workspace,
                   SolverStats *stats)
{
    ScalarField *alternate = &workspace->field;
    ScalarField *pressure_pipeline = &state->pressure_star;
    const uint64_t total_start = solver_time_ns();
    uint64_t kernel_start;

    /* Psi -> Phi_lower -> correction, alternating buffers without a copy. */
    kernel_start = solver_time_ns();
    backend_pressure_solve_x(grid, config, &state->velocity,
                             alternate, pressure_pipeline,
                             &workspace->scratch);
    stats->pressure_kernel_ns[DIRECTION_X] +=
        solver_time_ns() - kernel_start;

    kernel_start = solver_time_ns();
    backend_pressure_solve_y(grid, pressure_pipeline, alternate,
                             &workspace->scratch);
    stats->pressure_kernel_ns[DIRECTION_Y] +=
        solver_time_ns() - kernel_start;

    kernel_start = solver_time_ns();
    backend_pressure_solve_z(grid, alternate, pressure_pipeline,
                             &workspace->scratch);
    stats->pressure_kernel_ns[DIRECTION_Z] +=
        solver_time_ns() - kernel_start;

    kernel_start = solver_time_ns();
    pressure_finish_step(grid, &state->pressure, pressure_pipeline);
    stats->pressure_update_ns += solver_time_ns() - kernel_start;
    stats->pressure_total_ns += solver_time_ns() - total_start;
}
