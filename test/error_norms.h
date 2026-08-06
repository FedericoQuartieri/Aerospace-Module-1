#ifndef TEST_ERROR_NORMS_H
#define TEST_ERROR_NORMS_H

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "field.h"
#include "solver.h"

typedef struct ErrorNorms {
    Real L1;
    Real L2;
    Real Linf;
} ErrorNorms;

typedef struct SolverErrorNorms {
    ErrorNorms velocity_x;
    ErrorNorms velocity_y;
    ErrorNorms velocity_z;
    ErrorNorms pressure;
} SolverErrorNorms;

/*
 * Observed convergence rate between two discretizations:
 *
 *            log(error_coarse / error_fine)
 *   rate = ---------------------------------
 *              log(h_coarse / h_fine)
 *
 * h can represent either the spatial spacing or the time step.  The errors
 * must be positive and the two spacings must be positive and distinct.
 */
static inline Real compute_convergence_rate(Real error_coarse,
                                            Real error_fine,
                                            Real h_coarse,
                                            Real h_fine)
{
    if (error_coarse <= (Real)0 ||
        error_fine <= (Real)0 ||
        h_coarse <= (Real)0 ||
        h_fine <= (Real)0 ||
        h_coarse == h_fine) {
        return (Real)NAN;
    }

    return (Real)(log((double)(error_coarse / error_fine)) /
                  log((double)(h_coarse / h_fine)));
}

static inline ErrorNorms compute_error_norms(const Decomp *d,
                                             const Real *numerical,
                                             const Real *exact)
{
    ErrorNorms error = {0.0, 0.0, 0.0};
    const Real dV = (Real)DX * (Real)DY * (Real)DZ;

    for (int k = 0; k < d->n[2]; k++) {
        for (int j = 0; j < d->n[1]; j++) {
            size_t row = decomp_index(d, 0, j, k);

            for (int i = 0; i < d->n[0]; i++) {
                size_t index = row + (size_t)i;
                const Real difference =
                    (Real)fabs((double)(numerical[index] - exact[index]));

                error.L1 += difference;
                error.L2 += difference * difference;
                if (difference > error.Linf) {
                    error.Linf = difference;
                }
            }
        }
    }

    error.L1 *= dV;
    error.L2 = (Real)sqrt((double)(error.L2 * dV));

    return error;
}

/*
 * Pressure is defined up to an additive constant.  Remove the mean
 * difference between numerical and exact pressure before computing norms.
 */
static inline ErrorNorms compute_pressure_error_norms(const Decomp *d,
                                                      const Real *numerical,
                                                      const Real *exact)
{
    ErrorNorms error = {0.0, 0.0, 0.0};
    const Real dV = (Real)DX * (Real)DY * (Real)DZ;
    const size_t size = (size_t)d->n[0] * (size_t)d->n[1] * (size_t)d->n[2];
    Real mean_difference = 0.0;

    /* The constant to remove is a property of the whole field, so this sum
     * will have to become a global reduction once the grid is split. */
    for (int k = 0; k < d->n[2]; k++) {
        for (int j = 0; j < d->n[1]; j++) {
            size_t row = decomp_index(d, 0, j, k);
            for (int i = 0; i < d->n[0]; i++) {
                size_t index = row + (size_t)i;
                mean_difference += numerical[index] - exact[index];
            }
        }
    }
    mean_difference /= (Real)size;

    for (int k = 0; k < d->n[2]; k++) {
        for (int j = 0; j < d->n[1]; j++) {
            size_t row = decomp_index(d, 0, j, k);

            for (int i = 0; i < d->n[0]; i++) {
                size_t index = row + (size_t)i;
                const Real difference = (Real)fabs(
                    (double)(numerical[index] - exact[index] -
                             mean_difference));

                error.L1 += difference;
                error.L2 += difference * difference;
                if (difference > error.Linf) {
                    error.Linf = difference;
                }
            }
        }
    }

    error.L1 *= dV;
    error.L2 = (Real)sqrt((double)(error.L2 * dV));

    return error;
}

/* Unlike vectorField_fill, these take an already physical time. */
static inline void fill_exact_velocity(const Decomp *d,
                                       VectorField *exact,
                                       VectorFunction velocity_fn,
                                       Real time)
{
    for (int k = 0; k < d->n[2]; k++) {
        const int gk = decomp_global(d, k, 2);

        for (int j = 0; j < d->n[1]; j++) {
            const int gj = decomp_global(d, j, 1);
            const size_t row = decomp_index(d, 0, j, k);

            for (int i = 0; i < d->n[0]; i++) {
                const int gi = decomp_global(d, i, 0);
                const Real x = centered_physical_coord(gi, 0);
                const Real y = centered_physical_coord(gj, 1);
                const Real z = centered_physical_coord(gk, 2);
                const size_t index = row + (size_t)i;

                exact->v_x[index] =
                    velocity_fn(staggered_physical_coord(gi, 0),
                                y, z, time, 0);
                exact->v_y[index] =
                    velocity_fn(x, staggered_physical_coord(gj, 1),
                                z, time, 1);
                exact->v_z[index] =
                    velocity_fn(x, y,
                                staggered_physical_coord(gk, 2),
                                time, 2);
            }
        }
    }
}

static inline void fill_exact_pressure(const Decomp *d,
                                       ScalarField *exact,
                                       ScalarFunction pressure_fn,
                                       Real time)
{
    for (int k = 0; k < d->n[2]; k++) {
        const int gk = decomp_global(d, k, 2);

        for (int j = 0; j < d->n[1]; j++) {
            const int gj = decomp_global(d, j, 1);
            const size_t row = decomp_index(d, 0, j, k);

            for (int i = 0; i < d->n[0]; i++) {
                const int gi = decomp_global(d, i, 0);
                const Real x = centered_physical_coord(gi, 0);
                const Real y = centered_physical_coord(gj, 1);
                const Real z = centered_physical_coord(gk, 2);

                exact->v[row + (size_t)i] = pressure_fn(x, y, z, time);
            }
        }
    }
}

static inline SolverErrorNorms compute_solver_error_norms(
    const Decomp *d,
    const SolverMemState *solver_mem_state,
    const Data *data,
    Real velocity_time,
    Real pressure_time)
{
    VectorField exact_velocity;
    ScalarField exact_pressure;

    vectorField_alloc(d, &exact_velocity);
    scalarField_alloc(d, &exact_pressure);

    fill_exact_velocity(d, &exact_velocity, data->velocity_fn, velocity_time);
    fill_exact_pressure(d, &exact_pressure, data->pressure_fn, pressure_time);

    const SolverErrorNorms errors = {
        .velocity_x =
            compute_error_norms(d, solver_mem_state->u.v_x,
                                exact_velocity.v_x),
        .velocity_y =
            compute_error_norms(d, solver_mem_state->u.v_y,
                                exact_velocity.v_y),
        .velocity_z =
            compute_error_norms(d, solver_mem_state->u.v_z,
                                exact_velocity.v_z),
        .pressure =
            compute_pressure_error_norms(d, solver_mem_state->pressure.v,
                                         exact_pressure.v),
    };

    free(exact_velocity.v_x);
    free(exact_velocity.v_y);
    free(exact_velocity.v_z);
    free(exact_pressure.v);

    return errors;
}

static inline void print_solver_error_norms(const Decomp *d,
                                            const SolverErrorNorms *errors,
                                            Real velocity_time,
                                            Real pressure_time)
{
    printf("\nManufactured-solution error report:\n");
    printf("  Grid: %d x %d x %d\n",
           d->n_global[0], d->n_global[1], d->n_global[2]);
    printf("  dt: %.2e\n", (double)DT);
    printf("  T:  %.2e\n", (double)T);
    printf("  Velocity verification time: %.2e\n",
           (double)velocity_time);
    printf("  Pressure verification time: %.2e\n",
           (double)pressure_time);
    printf("  L2 error u_x: %.10e\n", (double)errors->velocity_x.L2);
    printf("  L2 error u_y: %.10e\n", (double)errors->velocity_y.L2);
    printf("  L2 error u_z: %.10e\n", (double)errors->velocity_z.L2);
    printf("  L2 error p:   %.10e\n", (double)errors->pressure.L2);
}

#endif
