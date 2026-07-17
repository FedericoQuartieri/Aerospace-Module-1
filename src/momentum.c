/* DESIGN: momentum orchestrates Eta -> Zeta -> U.  The backend owns each full
 * directional update; the shared workspace is reused component by component. */
#include "kernels.h"
#include "solver_internal.h"

void momentum_step(const Grid *grid,
                   const SolverConfig *config,
                   const ProblemDefinition *problem,
                   SolverState *state,
                   const ScalarField *gamma,
                   SolverWorkspace *workspace,
                   SolverStats *stats,
                   size_t timestep)
{
    uint64_t total_start = solver_time_ns();
    uint64_t kernel_start;
    Direction component;

    kernel_start = solver_time_ns();
    for (component = DIRECTION_X;
         component < DIRECTION_COUNT;
         component = (Direction)(component + 1)) {
        backend_momentum_solve_x(
            grid, config, problem,
            &state->eta.component[component],
            &state->zeta.component[component],
            &state->velocity.component[component],
            &state->pressure_star, gamma, component, timestep,
            &workspace->scratch);
    }
    stats->momentum_kernel_ns[DIRECTION_X] +=
        solver_time_ns() - kernel_start;

    kernel_start = solver_time_ns();
    for (component = DIRECTION_X;
         component < DIRECTION_COUNT;
         component = (Direction)(component + 1)) {
        backend_momentum_solve_y(
            grid, config, problem,
            &state->eta.component[component],
            &state->zeta.component[component],
            &workspace->field, gamma, component, timestep,
            &workspace->scratch);
    }
    stats->momentum_kernel_ns[DIRECTION_Y] +=
        solver_time_ns() - kernel_start;

    kernel_start = solver_time_ns();
    for (component = DIRECTION_X;
         component < DIRECTION_COUNT;
         component = (Direction)(component + 1)) {
        backend_momentum_solve_z(
            grid, config, problem,
            &state->zeta.component[component],
            &state->velocity.component[component],
            &workspace->field, gamma, component, timestep,
            &workspace->scratch);
    }
    stats->momentum_kernel_ns[DIRECTION_Z] +=
        solver_time_ns() - kernel_start;

    stats->momentum_total_ns += solver_time_ns() - total_start;
}
