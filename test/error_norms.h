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

static inline ErrorNorms compute_error_norms(const Real *numerical,
                                             const Real *exact,
                                             size_t size)
{
    ErrorNorms error = {0.0, 0.0, 0.0};
    const Real dV = (Real)DX * (Real)DY * (Real)DZ;

    for (size_t index = 0; index < size; index++) {
        const Real difference =
            (Real)fabs((double)(numerical[index] - exact[index]));

        error.L1 += difference;
        error.L2 += difference * difference;
        if (difference > error.Linf) {
            error.Linf = difference;
        }
    }

    error.L1 *= dV;
    error.L2 = (Real)sqrt((double)(error.L2 * dV));

    return error;
}

static inline void fill_exact_velocity(VectorField *exact,
                                       VectorFunction velocity_fn,
                                       Real time)
{
    size_t index = 0;

    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                const Real x = centered_physical_coord(i, 0);
                const Real y = centered_physical_coord(j, 1);
                const Real z = centered_physical_coord(k, 2);

                exact->v_x[index] =
                    velocity_fn(staggered_physical_coord(i, 0),
                                y, z, time, 0);
                exact->v_y[index] =
                    velocity_fn(x, staggered_physical_coord(j, 1),
                                z, time, 1);
                exact->v_z[index] =
                    velocity_fn(x, y,
                                staggered_physical_coord(k, 2),
                                time, 2);
                index++;
            }
        }
    }
}

static inline void fill_exact_pressure(ScalarField *exact,
                                       ScalarFunction pressure_fn,
                                       Real time)
{
    size_t index = 0;

    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                const Real x = centered_physical_coord(i, 0);
                const Real y = centered_physical_coord(j, 1);
                const Real z = centered_physical_coord(k, 2);

                exact->v[index] = pressure_fn(x, y, z, time);
                index++;
            }
        }
    }
}

static inline SolverErrorNorms compute_solver_error_norms(
    const SolverMemState *solver_mem_state,
    const Data *data,
    Real velocity_time,
    Real pressure_time)
{
    VectorField exact_velocity;
    ScalarField exact_pressure;

    vectorField_alloc(&exact_velocity);
    scalarField_alloc(&exact_pressure);

    fill_exact_velocity(&exact_velocity, data->velocity_fn, velocity_time);
    fill_exact_pressure(&exact_pressure, data->pressure_fn, pressure_time);

    const SolverErrorNorms errors = {
        .velocity_x =
            compute_error_norms(solver_mem_state->u.v_x,
                                exact_velocity.v_x, GRID_CELLS),
        .velocity_y =
            compute_error_norms(solver_mem_state->u.v_y,
                                exact_velocity.v_y, GRID_CELLS),
        .velocity_z =
            compute_error_norms(solver_mem_state->u.v_z,
                                exact_velocity.v_z, GRID_CELLS),
        .pressure =
            compute_error_norms(solver_mem_state->pressure.v,
                                exact_pressure.v, GRID_CELLS),
    };

    free(exact_velocity.v_x);
    free(exact_velocity.v_y);
    free(exact_velocity.v_z);
    free(exact_pressure.v);

    return errors;
}

static inline void print_solver_error_norms(const SolverErrorNorms *errors,
                                            Real velocity_time,
                                            Real pressure_time)
{
    printf("\nManufactured-solution error report:\n");
    printf("  Grid: %d x %d x %d\n", WIDTH, HEIGHT, DEPTH);
    printf("  dt: %.2e\n", (double)DT);
    printf("  T:  %.2e\n", (double)T);
    printf("  Velocity verification time: %.10e\n",
           (double)velocity_time);
    printf("  Pressure verification time: %.10e\n",
           (double)pressure_time);
    printf("  L2 error u_x: %.10e\n", (double)errors->velocity_x.L2);
    printf("  L2 error u_y: %.10e\n", (double)errors->velocity_y.L2);
    printf("  L2 error u_z: %.10e\n", (double)errors->velocity_z.L2);
    printf("  L2 error p:   %.10e\n", (double)errors->pressure.L2);
}

#endif
