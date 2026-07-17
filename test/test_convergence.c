#include "test_support.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static int spatial_study(const ManufacturedCase *test_case, bool verbose)
{
    static const size_t extents[] = {16, 32, 64};
    enum { LEVEL_COUNT = sizeof(extents) / sizeof(extents[0]) };
    ErrorReport report[LEVEL_COUNT];
    Real velocity_error[LEVEL_COUNT];
    Real pressure_error[LEVEL_COUNT];
    Real spacing[LEVEL_COUNT];
    size_t level;

    if (verbose) {
        printf("Spatial convergence: case=%s backend=%s\n",
               test_case->problem.name, solver_backend_name());
        printf("  N       h             velocity L2      pressure L2"
               "      divergence L2    boundary Linf\n");
    }
    for (level = 0; level < LEVEL_COUNT; ++level) {
        SolverConfig config = test_case->base_config;
        config.extent[DIRECTION_X] = extents[level];
        config.extent[DIRECTION_Y] = extents[level];
        config.extent[DIRECTION_Z] = extents[level];
        config.dt = (Real)0.0007;
        config.steps = 10;
        spacing[level] =
            (Real)2 * config.domain_length[DIRECTION_X] /
            (Real)(2 * extents[level] - 1);
        if (!run_manufactured_case(test_case, &config, &report[level])) {
            fprintf(stderr, "spatial solve failed at %zux%zux%zu\n",
                    extents[level], extents[level], extents[level]);
            return 1;
        }
        velocity_error[level] = combined_velocity_l2(&report[level]);
        pressure_error[level] = report[level].pressure.l2;
        if (verbose) {
            printf("  %-3zu  %.9e  %.9e  %.9e  %.9e  %.9e\n",
                   extents[level], (double)spacing[level],
                   (double)velocity_error[level],
                   (double)pressure_error[level],
                   (double)report[level].divergence_l2,
                   (double)report[level].boundary_linf);
        }
    }

    if (!(velocity_error[2] < velocity_error[1] &&
          pressure_error[2] < pressure_error[1])) {
        fprintf(stderr, "spatial errors did not decrease on 32->64\n");
        return 1;
    }
    {
        const Real velocity_order = convergence_order(
            velocity_error[1], velocity_error[2], spacing[1], spacing[2]);
        const Real pressure_order = convergence_order(
            pressure_error[1], pressure_error[2], spacing[1], spacing[2]);
        if (verbose) {
            printf("  order 32->64: velocity=%.6f (min %.2f), "
                   "pressure=%.6f (min %.2f)\n",
                   (double)velocity_order,
                   (double)test_case->min_velocity_space_order,
                   (double)pressure_order,
                   (double)test_case->min_pressure_space_order);
        }
        if (velocity_order < test_case->min_velocity_space_order ||
            pressure_order < test_case->min_pressure_space_order) {
            fprintf(stderr,
                    "spatial order failed: velocity %.6f (min %.6f), "
                    "pressure %.6f (min %.6f)\n",
                    (double)velocity_order,
                    (double)test_case->min_velocity_space_order,
                    (double)pressure_order,
                    (double)test_case->min_pressure_space_order);
            return 1;
        }
    }
    return 0;
}

static int temporal_study(const ManufacturedCase *test_case, bool verbose)
{
    static const Real dt[] = {
        (Real)0.1, (Real)0.05, (Real)0.025,
        (Real)0.0125, (Real)0.00625
    };
    enum { LEVEL_COUNT = sizeof(dt) / sizeof(dt[0]) };
    enum { TEMPORAL_GRID_EXTENT = 64 };
    ErrorReport report[LEVEL_COUNT];
    Real velocity_error[LEVEL_COUNT];
    Real pressure_error[LEVEL_COUNT];
    size_t level;

    if (verbose) {
        printf("Temporal convergence: case=%s backend=%s grid=%d^3 "
               "final_time=0.5\n",
               test_case->problem.name, solver_backend_name(),
               TEMPORAL_GRID_EXTENT);
        printf("  dt           steps  velocity L2      pressure L2"
               "      divergence L2    boundary Linf\n");
    }
    for (level = 0; level < LEVEL_COUNT; ++level) {
        SolverConfig config = test_case->base_config;
        config.extent[DIRECTION_X] = TEMPORAL_GRID_EXTENT;
        config.extent[DIRECTION_Y] = TEMPORAL_GRID_EXTENT;
        config.extent[DIRECTION_Z] = TEMPORAL_GRID_EXTENT;
        config.dt = dt[level];
        config.steps = (size_t)((Real)0.5 / dt[level] + (Real)0.5);
        if (!run_manufactured_case(test_case, &config, &report[level])) {
            fprintf(stderr, "temporal solve failed for dt %.9g\n",
                    (double)dt[level]);
            return 1;
        }
        velocity_error[level] = combined_velocity_l2(&report[level]);
        pressure_error[level] = report[level].pressure.l2;
        if (verbose) {
            printf("  %.7f  %-5zu  %.9e  %.9e  %.9e  %.9e\n",
                   (double)dt[level], config.steps,
                   (double)velocity_error[level],
                   (double)pressure_error[level],
                   (double)report[level].divergence_l2,
                   (double)report[level].boundary_linf);
        }
    }

    for (level = 1; level <= 2; ++level) {
        const Real velocity_order = convergence_order(
            velocity_error[level], velocity_error[level + 1],
            dt[level], dt[level + 1]);
        const Real pressure_order = convergence_order(
            pressure_error[level], pressure_error[level + 1],
            dt[level], dt[level + 1]);
        if (verbose) {
            printf("  order %.6g->%.6g: velocity=%.6f (min %.2f), "
                   "pressure=%.6f (min %.2f)\n",
                   (double)dt[level], (double)dt[level + 1],
                   (double)velocity_order,
                   (double)test_case->min_velocity_time_order,
                   (double)pressure_order,
                   (double)test_case->min_pressure_time_order);
        }
        if (velocity_order < test_case->min_velocity_time_order ||
            pressure_order < test_case->min_pressure_time_order) {
            fprintf(stderr,
                    "temporal order failed for %.6g->%.6g: "
                    "velocity %.6f (min %.6f), pressure %.6f (min %.6f)\n",
                    (double)dt[level], (double)dt[level + 1],
                    (double)velocity_order,
                    (double)test_case->min_velocity_time_order,
                    (double)pressure_order,
                    (double)test_case->min_pressure_time_order);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    const ManufacturedCase *test_case = &MANUFACTURED_CASES[0];
    bool run_spatial = false;
    bool run_temporal = false;
    bool mode_selected = false;
    bool verbose = false;
    int argument;

    for (argument = 1; argument < argc; ++argument) {
        if (strcmp(argv[argument], "--spatial") == 0) {
            run_spatial = true;
            mode_selected = true;
        } else if (strcmp(argv[argument], "--temporal") == 0) {
            run_temporal = true;
            mode_selected = true;
        } else if (strcmp(argv[argument], "--verbose") == 0) {
            verbose = true;
        } else {
            fprintf(stderr,
                    "usage: %s [--spatial] [--temporal] [--verbose]\n",
                    argv[0]);
            return 1;
        }
    }
    if (!mode_selected) {
        run_spatial = true;
        run_temporal = true;
    }
    if (run_spatial && spatial_study(test_case, verbose) != 0) {
        return 1;
    }
    if (run_temporal && temporal_study(test_case, verbose) != 0) {
        return 1;
    }
    return 0;
}
