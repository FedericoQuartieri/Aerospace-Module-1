#ifndef TEST_SUPPORT_H
#define TEST_SUPPORT_H

#include "manufactured_cases.h"

typedef struct {
    Real l2;
    Real linf;
} ErrorNorm;

/* Pressure norms are evaluated at the half timestep after removing the mean
 * offset; velocity and boundary norms use the completed integer timestep. */
typedef struct {
    ErrorNorm velocity[DIRECTION_COUNT];
    ErrorNorm pressure;
    Real divergence_l2;
    Real boundary_linf;
} ErrorReport;

bool run_manufactured_case(const ManufacturedCase *test_case,
                           const SolverConfig *config,
                           ErrorReport *report);
Real combined_velocity_l2(const ErrorReport *report);
Real convergence_order(Real coarse_error,
                       Real fine_error,
                       Real coarse_scale,
                       Real fine_scale);

#endif
