#include "test_support.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static bool report_within_limits(const ManufacturedCase *test_case,
                                 const ErrorReport *report)
{
    Direction component;
    for (component = DIRECTION_X;
         component < DIRECTION_COUNT;
         component = (Direction)(component + 1)) {
        if (report->velocity[component].l2 >
                test_case->max_velocity_l2 ||
            report->velocity[component].linf >
                test_case->max_velocity_linf) {
            return false;
        }
    }
    return report->pressure.l2 <= test_case->max_pressure_l2 &&
           report->pressure.linf <= test_case->max_pressure_linf &&
           report->divergence_l2 <= test_case->max_divergence_l2 &&
           report->boundary_linf <= test_case->max_boundary_linf;
}

static void print_failure(const ManufacturedCase *test_case,
                          const ErrorReport *report)
{
    fprintf(stderr,
            "%s failed on backend %s: velocity L2=(%.9e, %.9e, %.9e), "
            "Linf=(%.9e, %.9e, %.9e), pressure=(%.9e, %.9e), "
            "divergence=%.9e, boundary=%.9e\n",
            test_case->problem.name, solver_backend_name(),
            (double)report->velocity[DIRECTION_X].l2,
            (double)report->velocity[DIRECTION_Y].l2,
            (double)report->velocity[DIRECTION_Z].l2,
            (double)report->velocity[DIRECTION_X].linf,
            (double)report->velocity[DIRECTION_Y].linf,
            (double)report->velocity[DIRECTION_Z].linf,
            (double)report->pressure.l2,
            (double)report->pressure.linf,
            (double)report->divergence_l2,
            (double)report->boundary_linf);
}

static void print_report(const ManufacturedCase *test_case,
                         const ErrorReport *report)
{
    const SolverConfig *config = &test_case->base_config;
    printf("[PASS] case=%s backend=%s grid=%zux%zux%zu dt=%.9g steps=%zu\n",
           test_case->problem.name, solver_backend_name(),
           config->extent[DIRECTION_X],
           config->extent[DIRECTION_Y],
           config->extent[DIRECTION_Z],
           (double)config->dt, config->steps);
    printf("  velocity L2   = (%.9e, %.9e, %.9e)\n",
           (double)report->velocity[DIRECTION_X].l2,
           (double)report->velocity[DIRECTION_Y].l2,
           (double)report->velocity[DIRECTION_Z].l2);
    printf("  velocity Linf = (%.9e, %.9e, %.9e)\n",
           (double)report->velocity[DIRECTION_X].linf,
           (double)report->velocity[DIRECTION_Y].linf,
           (double)report->velocity[DIRECTION_Z].linf);
    printf("  pressure L2/Linf = %.9e / %.9e\n",
           (double)report->pressure.l2,
           (double)report->pressure.linf);
    printf("  divergence L2 = %.9e, boundary Linf = %.9e\n",
           (double)report->divergence_l2,
           (double)report->boundary_linf);
}

static void print_usage(const char *program)
{
    fprintf(stderr, "usage: %s [--verbose]\n", program);
}

int main(int argc, char **argv)
{
    bool verbose = false;
    size_t case_index;

    if (argc == 2 && strcmp(argv[1], "--verbose") == 0) {
        verbose = true;
    } else if (argc != 1) {
        print_usage(argv[0]);
        return 1;
    }

    for (case_index = 0;
         case_index < MANUFACTURED_CASE_COUNT;
         ++case_index) {
        const ManufacturedCase *test_case =
            &MANUFACTURED_CASES[case_index];
        ErrorReport report;
        const bool solved = run_manufactured_case(
            test_case, &test_case->base_config, &report);

        if (!solved) {
            fprintf(stderr, "%s solve failed on backend %s\n",
                    test_case->problem.name, solver_backend_name());
            return 1;
        }
        if (!report_within_limits(test_case, &report)) {
            print_failure(test_case, &report);
            return 1;
        }
        if (verbose) {
            print_report(test_case, &report);
        }
    }
    return 0;
}
