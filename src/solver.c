/* DESIGN: Solver is the sole memory owner.  Initialization allocates every
 * field and scratch buffer; solver_step only orchestrates numerical work. */
#include "kernels.h"
#include "solver_internal.h"

#include <math.h>
#include <string.h>
#include <time.h>

#ifndef SOLVER_DEFAULT_NX
#define SOLVER_DEFAULT_NX 128
#endif
#ifndef SOLVER_DEFAULT_NY
#define SOLVER_DEFAULT_NY 128
#endif
#ifndef SOLVER_DEFAULT_NZ
#define SOLVER_DEFAULT_NZ 128
#endif
#ifndef SOLVER_DEFAULT_DT
#define SOLVER_DEFAULT_DT 0.001
#endif
#ifndef SOLVER_DEFAULT_TOTAL_TIME
#define SOLVER_DEFAULT_TOTAL_TIME 0.01
#endif
#ifndef SOLVER_DEFAULT_OUTPUT_FREQUENCY
#define SOLVER_DEFAULT_OUTPUT_FREQUENCY 0
#endif

uint64_t solver_time_ns(void)
{
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (uint64_t)time.tv_sec * UINT64_C(1000000000) +
           (uint64_t)time.tv_nsec;
}

SolverConfig solver_default_config(void)
{
    static const Real pi =
        (Real)3.141592653589793238462643383279502884;
    SolverConfig config;
    const Real total_time = (Real)SOLVER_DEFAULT_TOTAL_TIME;

    memset(&config, 0, sizeof(config));
    config.extent[DIRECTION_X] = SOLVER_DEFAULT_NX;
    config.extent[DIRECTION_Y] = SOLVER_DEFAULT_NY;
    config.extent[DIRECTION_Z] = SOLVER_DEFAULT_NZ;
    config.domain_length[DIRECTION_X] = pi;
    config.domain_length[DIRECTION_Y] = pi;
    config.domain_length[DIRECTION_Z] = pi;
    config.dt = (Real)SOLVER_DEFAULT_DT;
    config.steps = (size_t)(total_time / config.dt + (Real)1e-12);
    config.viscosity = (Real)1;
    config.output_frequency = SOLVER_DEFAULT_OUTPUT_FREQUENCY;
    config.output_directory = "output";
    return config;
}

static bool valid_config(const SolverConfig *config,
                         const ProblemDefinition *problem)
{
    Direction direction;

    if (config == NULL || problem == NULL ||
        problem->initial_velocity == NULL ||
        problem->initial_pressure == NULL ||
        problem->boundary_velocity == NULL ||
        problem->forcing == NULL ||
        problem->permeability == NULL ||
        !isfinite(config->dt) || config->dt <= (Real)0 ||
        config->steps == 0 || !isfinite(config->viscosity) ||
        config->viscosity <= (Real)0) {
        return false;
    }
    for (direction = DIRECTION_X;
         direction < DIRECTION_COUNT;
         direction = (Direction)(direction + 1)) {
        if (config->extent[direction] < 2 ||
            config->extent[direction] > (SIZE_MAX - 1) / 2 ||
            !isfinite(config->domain_length[direction]) ||
            config->domain_length[direction] <= (Real)0) {
            return false;
        }
    }
    if (config->output_frequency > 0 &&
        (config->output_directory == NULL ||
         config->output_directory[0] == '\0')) {
        return false;
    }
    return true;
}

static bool initialize_state(Solver *solver)
{
    const size_t count = solver->grid.cell_count;
    Direction component;
    size_t i;
    size_t j;
    size_t k;

    if (!vector_field_init(&solver->state.eta, count) ||
        !vector_field_init(&solver->state.zeta, count) ||
        !vector_field_init(&solver->state.velocity, count) ||
        !scalar_field_init(&solver->state.pressure, count) ||
        !scalar_field_init(&solver->state.pressure_star, count) ||
        !scalar_field_init(&solver->gamma, count) ||
        !scalar_field_init(&solver->workspace.field, count) ||
        !real_buffer_init(&solver->workspace.scratch,
                          backend_scratch_capacity(&solver->grid))) {
        return false;
    }

    for (k = 0; k < solver->grid.extent[DIRECTION_Z]; ++k) {
        for (j = 0; j < solver->grid.extent[DIRECTION_Y]; ++j) {
            for (i = 0; i < solver->grid.extent[DIRECTION_X]; ++i) {
                const size_t index = grid_index(&solver->grid, i, j, k);
                const Real px = grid_pressure_coordinate(
                    &solver->grid, DIRECTION_X, i);
                const Real py = grid_pressure_coordinate(
                    &solver->grid, DIRECTION_Y, j);
                const Real pz = grid_pressure_coordinate(
                    &solver->grid, DIRECTION_Z, k);
                const Real permeability = solver->problem->permeability(
                    px, py, pz, (Real)0);
                Real beta;

                if (!isfinite(permeability) || permeability <= (Real)0) {
                    return false;
                }
                beta = (Real)1 +
                       solver->config.dt * solver->config.viscosity /
                           ((Real)2 * permeability);
                solver->gamma.data[index] =
                    solver->config.dt * solver->config.viscosity /
                    ((Real)2 * beta);
                solver->state.pressure.data[index] =
                    solver->problem->initial_pressure(px, py, pz, (Real)0);

                for (component = DIRECTION_X;
                     component < DIRECTION_COUNT;
                     component = (Direction)(component + 1)) {
                    const Real x = grid_velocity_coordinate(
                        &solver->grid, DIRECTION_X, component, i);
                    const Real y = grid_velocity_coordinate(
                        &solver->grid, DIRECTION_Y, component, j);
                    const Real z = grid_velocity_coordinate(
                        &solver->grid, DIRECTION_Z, component, k);
                    const Real initial = solver->problem->initial_velocity(
                        x, y, z, (Real)0, component);
                    solver->state.eta.component[component].data[index] = initial;
                    solver->state.zeta.component[component].data[index] = initial;
                    solver->state.velocity.component[component].data[index] =
                        initial;
                }
            }
        }
    }

    /* Baseline startup: the first predictor is an exact copy of p(t=0). */
    scalar_field_copy(&solver->state.pressure_star,
                      &solver->state.pressure);
    return true;
}

SolverStatus solver_init(Solver *solver,
                         const SolverConfig *config,
                         const ProblemDefinition *problem)
{
    uint64_t start;

    if (solver == NULL) {
        return SOLVER_INVALID_CONFIG;
    }
    memset(solver, 0, sizeof(*solver));
    start = solver_time_ns();
    if (!valid_config(config, problem)) {
        return SOLVER_INVALID_CONFIG;
    }
    solver->config = *config;
    solver->problem = problem;
    if (!grid_init(&solver->grid, config)) {
        solver_destroy(solver);
        return SOLVER_INVALID_CONFIG;
    }
    if (!initialize_state(solver)) {
        solver_destroy(solver);
        return SOLVER_NUMERICAL_ERROR;
    }
    if (!output_writer_init(&solver->output, config)) {
        solver_destroy(solver);
        return SOLVER_OUTPUT_ERROR;
    }
    solver->stats.init_ns = solver_time_ns() - start;
    return SOLVER_SUCCESS;
}

static bool scalar_field_is_finite(const ScalarField *field)
{
    size_t q;
    for (q = 0; q < field->count; ++q) {
        if (!isfinite(field->data[q])) {
            return false;
        }
    }
    return true;
}

static bool solver_fields_are_finite(const Solver *solver)
{
    Direction component;
    for (component = DIRECTION_X;
         component < DIRECTION_COUNT;
         component = (Direction)(component + 1)) {
        if (!scalar_field_is_finite(&solver->state.eta.component[component]) ||
            !scalar_field_is_finite(&solver->state.zeta.component[component]) ||
            !scalar_field_is_finite(
                &solver->state.velocity.component[component])) {
            return false;
        }
    }
    return scalar_field_is_finite(&solver->state.pressure) &&
           scalar_field_is_finite(&solver->state.pressure_star) &&
           scalar_field_is_finite(&solver->gamma) &&
           scalar_field_is_finite(&solver->workspace.field);
}

SolverStatus solver_step(Solver *solver, size_t timestep)
{
    const uint64_t start = solver_time_ns();
    momentum_step(&solver->grid, &solver->config, solver->problem,
                  &solver->state, &solver->gamma, &solver->workspace,
                  &solver->stats, timestep);
    pressure_step(&solver->grid, &solver->config, &solver->state,
                  &solver->workspace, &solver->stats);
    solver->stats.timestep_compute_ns += solver_time_ns() - start;
    return solver_fields_are_finite(solver)
        ? SOLVER_SUCCESS
        : SOLVER_NUMERICAL_ERROR;
}

SolverStatus solver_solve(Solver *solver)
{
    size_t timestep;
    if (solver == NULL || solver->problem == NULL) {
        return SOLVER_INVALID_CONFIG;
    }
    for (timestep = 1; timestep <= solver->config.steps; ++timestep) {
        SolverStatus status = solver_step(solver, timestep);
        uint64_t output_start;
        if (status != SOLVER_SUCCESS) {
            return status;
        }
        ++solver->stats.completed_steps;
        if (!solver->output.enabled ||
            timestep % solver->output.frequency != 0) {
            continue;
        }
        output_start = solver_time_ns();
        if (!output_writer_write(
                &solver->output, &solver->grid, timestep,
                (Real)timestep * solver->config.dt,
                ((Real)timestep - (Real)0.5) * solver->config.dt,
                &solver->state.velocity, &solver->state.pressure)) {
            solver->stats.output_ns += solver_time_ns() - output_start;
            return SOLVER_OUTPUT_ERROR;
        }
        solver->stats.output_ns += solver_time_ns() - output_start;
    }
    return SOLVER_SUCCESS;
}

void solver_destroy(Solver *solver)
{
    if (solver == NULL) {
        return;
    }
    output_writer_destroy(&solver->output);
    real_buffer_destroy(&solver->workspace.scratch);
    scalar_field_destroy(&solver->workspace.field);
    scalar_field_destroy(&solver->gamma);
    scalar_field_destroy(&solver->state.pressure_star);
    scalar_field_destroy(&solver->state.pressure);
    vector_field_destroy(&solver->state.velocity);
    vector_field_destroy(&solver->state.zeta);
    vector_field_destroy(&solver->state.eta);
    memset(solver, 0, sizeof(*solver));
}

const char *solver_backend_name(void)
{
    return SOLVER_BACKEND_NAME;
}

static double average_ms(uint64_t total_ns, size_t steps)
{
    return steps == 0
        ? 0.0
        : (double)total_ns / (double)steps / 1.0e6;
}

void solver_print_stats(const Solver *solver, FILE *stream)
{
    const size_t steps = solver->stats.completed_steps;
    fprintf(stream, "Backend: %s\n", solver_backend_name());
    fprintf(stream, "Completed steps: %zu\n", steps);
    fprintf(stream, "Initialization: %.3f ms\n",
            (double)solver->stats.init_ns / 1.0e6);
    fprintf(stream, "Momentum X/Y/Z mean: %.3f / %.3f / %.3f ms\n",
            average_ms(solver->stats.momentum_kernel_ns[DIRECTION_X], steps),
            average_ms(solver->stats.momentum_kernel_ns[DIRECTION_Y], steps),
            average_ms(solver->stats.momentum_kernel_ns[DIRECTION_Z], steps));
    fprintf(stream, "Momentum total mean: %.3f ms\n",
            average_ms(solver->stats.momentum_total_ns, steps));
    fprintf(stream, "Pressure X/Y/Z mean: %.3f / %.3f / %.3f ms\n",
            average_ms(solver->stats.pressure_kernel_ns[DIRECTION_X], steps),
            average_ms(solver->stats.pressure_kernel_ns[DIRECTION_Y], steps),
            average_ms(solver->stats.pressure_kernel_ns[DIRECTION_Z], steps));
    fprintf(stream, "Pressure update mean: %.3f ms\n",
            average_ms(solver->stats.pressure_update_ns, steps));
    fprintf(stream, "Pressure total mean: %.3f ms\n",
            average_ms(solver->stats.pressure_total_ns, steps));
    fprintf(stream, "Timestep compute mean: %.3f ms\n",
            average_ms(solver->stats.timestep_compute_ns, steps));
    fprintf(stream, "Output total: %.3f ms\n",
            (double)solver->stats.output_ns / 1.0e6);
}
