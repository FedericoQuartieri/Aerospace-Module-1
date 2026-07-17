#include "manufactured_cases.h"
#include "solver_internal.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    WORKLOAD_PAPER,
    WORKLOAD_SYNTHETIC
} BenchmarkWorkload;

static Real synthetic_velocity(Real x,
                               Real y,
                               Real z,
                               Real time,
                               Direction component)
{
    const Real scale = (Real)(component + 1);
    return scale * ((Real)0.2 + (Real)0.03 * x - (Real)0.02 * y +
                    (Real)0.01 * z + (Real)0.005 * time);
}

static Real synthetic_pressure(Real x, Real y, Real z, Real time)
{
    return (Real)0.02 * x - (Real)0.015 * y + (Real)0.01 * z +
           (Real)0.003 * time;
}

static Real synthetic_forcing(Real x,
                              Real y,
                              Real z,
                              Real time,
                              Direction component)
{
    const Real scale = (Real)(component + 1);
    return scale * ((Real)0.11 + (Real)0.013 * x + (Real)0.017 * y -
                    (Real)0.019 * z + (Real)0.007 * time);
}

static Real synthetic_permeability(Real x, Real y, Real z, Real time)
{
    (void)time;
    return (Real)1 + (Real)0.01 * (x + (Real)2 * y + (Real)3 * z);
}

static const ProblemDefinition SYNTHETIC_PROBLEM = {
    "synthetic-lightweight",
    synthetic_velocity,
    synthetic_pressure,
    synthetic_velocity,
    synthetic_forcing,
    synthetic_permeability
};

static bool parse_size(const char *text, size_t *value)
{
    char *end;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0) {
        return false;
    }
    *value = (size_t)parsed;
    return (unsigned long long)*value == parsed;
}

static bool parse_arguments(int argc,
                            char **argv,
                            size_t *extent,
                            size_t *warmup_steps,
                            size_t *measured_steps,
                            BenchmarkWorkload *workload)
{
    int argument;
    for (argument = 1; argument < argc; argument += 2) {
        if (argument + 1 >= argc) return false;
        if (strcmp(argv[argument], "--grid") == 0) {
            if (!parse_size(argv[argument + 1], extent)) return false;
        } else if (strcmp(argv[argument], "--warmup") == 0) {
            if (!parse_size(argv[argument + 1], warmup_steps)) return false;
        } else if (strcmp(argv[argument], "--steps") == 0) {
            if (!parse_size(argv[argument + 1], measured_steps)) return false;
        } else if (strcmp(argv[argument], "--workload") == 0) {
            if (strcmp(argv[argument + 1], "paper") == 0) {
                *workload = WORKLOAD_PAPER;
            } else if (strcmp(argv[argument + 1], "synthetic") == 0) {
                *workload = WORKLOAD_SYNTHETIC;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }
    return true;
}

static double ns_per_step_cell(uint64_t nanoseconds,
                               size_t steps,
                               size_t cells)
{
    return (double)nanoseconds / (double)steps / (double)cells;
}

int main(int argc, char **argv)
{
    size_t extent = 64;
    size_t warmup_steps = 2;
    size_t measured_steps = 10;
    BenchmarkWorkload workload = WORKLOAD_PAPER;
    SolverConfig config = MANUFACTURED_CASES[0].base_config;
    const ProblemDefinition *problem;
    Solver solver = {0};
    SolverStatus status;
    size_t step;

    if (!parse_arguments(argc, argv, &extent, &warmup_steps,
                         &measured_steps, &workload)) {
        fprintf(stderr,
                "usage: %s [--grid N] [--warmup N] [--steps N] "
                "[--workload paper|synthetic]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    config.extent[DIRECTION_X] = extent;
    config.extent[DIRECTION_Y] = extent;
    config.extent[DIRECTION_Z] = extent;
    config.steps = warmup_steps + measured_steps;
    config.output_frequency = 0;
    config.output_directory = NULL;
    problem = workload == WORKLOAD_PAPER
        ? &MANUFACTURED_CASES[0].problem
        : &SYNTHETIC_PROBLEM;

    /* Warmup advances the real solver state but is excluded from samples.
     * Output stays disabled so the reported time is compute-only. */
    status = solver_init(&solver, &config, problem);
    for (step = 1; status == SOLVER_SUCCESS && step <= warmup_steps; ++step) {
        status = solver_step(&solver, step);
    }
    if (status == SOLVER_SUCCESS) {
        const uint64_t init_ns = solver.stats.init_ns;
        /* Reset only accumulated solve timings; the measured steps continue
         * from the warmed state and therefore exercise normal timesteps. */
        memset(&solver.stats, 0, sizeof(solver.stats));
        solver.stats.init_ns = init_ns;
    }
    for (step = warmup_steps + 1;
         status == SOLVER_SUCCESS &&
         step <= warmup_steps + measured_steps;
         ++step) {
        status = solver_step(&solver, step);
        if (status == SOLVER_SUCCESS) ++solver.stats.completed_steps;
    }

    if (status != SOLVER_SUCCESS ||
        solver.stats.completed_steps != measured_steps) {
        fprintf(stderr, "benchmark solve failed with status %d\n", (int)status);
        solver_destroy(&solver);
        return EXIT_FAILURE;
    }

    printf("backend,workload,extent,warmup,steps,momentum_x_ns_per_cell,"
           "pressure_x_ns_per_cell,timestep_ns_per_cell\n");
    printf("%s,%s,%zu,%zu,%zu,%.9f,%.9f,%.9f\n",
           solver_backend_name(),
           workload == WORKLOAD_PAPER ? "paper" : "synthetic",
           extent, warmup_steps, measured_steps,
           ns_per_step_cell(
               solver.stats.momentum_kernel_ns[DIRECTION_X],
               measured_steps, solver.grid.cell_count),
           ns_per_step_cell(
               solver.stats.pressure_kernel_ns[DIRECTION_X],
               measured_steps, solver.grid.cell_count),
           ns_per_step_cell(solver.stats.timestep_compute_ns,
                            measured_steps, solver.grid.cell_count));

    solver_destroy(&solver);
    return EXIT_SUCCESS;
}
