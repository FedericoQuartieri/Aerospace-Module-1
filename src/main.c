#include "solver.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static Real paper_velocity(Real x,
                           Real y,
                           Real z,
                           Real time,
                           Direction component)
{
    switch (component) {
        case DIRECTION_X: return sin(x) * cos(time + y) * sin(z);
        case DIRECTION_Y: return cos(x) * sin(time + y) * sin(z);
        case DIRECTION_Z:
            return (Real)2 * cos(x) * cos(time + y) * cos(z);
        default: return (Real)0;
    }
}

static Real paper_pressure(Real x, Real y, Real z, Real time)
{
    return (Real)-3 * cos(x) * cos(time + y) * cos(z);
}

static Real paper_forcing(Real x,
                          Real y,
                          Real z,
                          Real time,
                          Direction component)
{
    const Real velocity = paper_velocity(x, y, z, time, component);
    Real time_derivative;
    Real pressure_gradient;

    switch (component) {
        case DIRECTION_X:
            time_derivative = -sin(x) * sin(time + y) * sin(z);
            pressure_gradient = (Real)3 * sin(x) * cos(time + y) * cos(z);
            break;
        case DIRECTION_Y:
            time_derivative = cos(x) * cos(time + y) * sin(z);
            pressure_gradient = (Real)3 * cos(x) * sin(time + y) * cos(z);
            break;
        case DIRECTION_Z:
            time_derivative =
                (Real)-2 * cos(x) * sin(time + y) * cos(z);
            pressure_gradient =
                (Real)3 * cos(x) * cos(time + y) * sin(z);
            break;
        default:
            return (Real)0;
    }
    return time_derivative + (Real)4 * velocity + pressure_gradient;
}

static Real unit_permeability(Real x, Real y, Real z, Real time)
{
    (void)x;
    (void)y;
    (void)z;
    (void)time;
    return (Real)1;
}

static const ProblemDefinition PAPER_PROBLEM = {
    "paper manufactured solution",
    paper_velocity,
    paper_pressure,
    paper_velocity,
    paper_forcing,
    unit_permeability
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

static bool parse_real(const char *text, Real *value)
{
    char *end;
    double parsed;
    errno = 0;
    parsed = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' ||
        !isfinite(parsed) || parsed <= 0) {
        return false;
    }
    *value = (Real)parsed;
    return true;
}

static bool parse_config(int argc, char **argv, SolverConfig *config)
{
    int argument;
    for (argument = 1; argument < argc; argument += 2) {
        if (argument + 1 >= argc) {
            return false;
        }
        if (strcmp(argv[argument], "--grid") == 0) {
            size_t extent;
            if (!parse_size(argv[argument + 1], &extent)) return false;
            config->extent[DIRECTION_X] = extent;
            config->extent[DIRECTION_Y] = extent;
            config->extent[DIRECTION_Z] = extent;
        } else if (strcmp(argv[argument], "--dt") == 0) {
            if (!parse_real(argv[argument + 1], &config->dt)) return false;
        } else if (strcmp(argv[argument], "--steps") == 0) {
            if (!parse_size(argv[argument + 1], &config->steps)) return false;
        } else if (strcmp(argv[argument], "--output-frequency") == 0) {
            if (strcmp(argv[argument + 1], "0") == 0) {
                config->output_frequency = 0;
            } else if (!parse_size(argv[argument + 1],
                                   &config->output_frequency)) {
                return false;
            }
        } else if (strcmp(argv[argument], "--output-directory") == 0) {
            config->output_directory = argv[argument + 1];
        } else {
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    SolverConfig config = solver_default_config();
    Solver solver = {0};
    SolverStatus status;

    if (!parse_config(argc, argv, &config)) {
        fprintf(stderr,
                "usage: %s [--grid N] [--dt DT] [--steps N] "
                "[--output-frequency N] [--output-directory PATH]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    status = solver_init(&solver, &config, &PAPER_PROBLEM);
    if (status == SOLVER_SUCCESS) {
        status = solver_solve(&solver);
    }
    if (status == SOLVER_SUCCESS) {
        solver_print_stats(&solver, stdout);
    } else {
        fprintf(stderr, "solver failed with status %d\n", (int)status);
    }
    solver_destroy(&solver);
    return status == SOLVER_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
