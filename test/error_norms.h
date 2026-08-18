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

static inline Real compute_convergence_rate(Real error_coarse,
                                            Real error_fine,
                                            Real h_coarse,
                                            Real h_fine)
{
    if (error_coarse <= (Real)0 || error_fine <= (Real)0 ||
        h_coarse <= (Real)0 || h_fine <= (Real)0 ||
        h_coarse == h_fine) {
        return (Real)NAN;
    }

    return (Real)(log((double)(error_coarse / error_fine)) /
                  log((double)(h_coarse / h_fine)));
}

static inline ErrorNorms reduce_error_norms(Real local_l1, Real local_l2,
                                            Real local_linf,
                                            const Domain *domain)
{
    Real local_sum[2] = {local_l1, local_l2};
    Real global_sum[2];
    Real global_linf;
    const Real dV = (Real)DX * (Real)DY * (Real)DZ;

    MPI_Allreduce(local_sum, global_sum, 2, mpi_real_type(), MPI_SUM,
                  domain->cart);
    MPI_Allreduce(&local_linf, &global_linf, 1, mpi_real_type(), MPI_MAX,
                  domain->cart);

    ErrorNorms result = {
        .L1 = global_sum[0] * dV,
        .L2 = (Real)sqrt((double)(global_sum[1] * dV)),
        .Linf = global_linf,
    };
    return result;
}

static inline ErrorNorms compute_error_norms(const Real *numerical,
                                             const Real *exact,
                                             const Domain *domain)
{
    Real local_l1 = 0;
    Real local_l2 = 0;
    Real local_linf = 0;

    for (int k = 0; k < domain->local[AXIS_Z]; k++) {
        for (int j = 0; j < domain->local[AXIS_Y]; j++) {
            for (int i = 0; i < domain->local[AXIS_X]; i++) {
                size_t index = domain_index(domain, i, j, k);
                Real difference = (Real)fabs(
                    (double)(numerical[index] - exact[index]));

                local_l1 += difference;
                local_l2 += difference * difference;
                if (difference > local_linf) local_linf = difference;
            }
        }
    }
    return reduce_error_norms(local_l1, local_l2, local_linf, domain);
}

static inline ErrorNorms compute_pressure_error_norms(
    const Real *numerical, const Real *exact, const Domain *domain)
{
    Real local_mean_sum = 0;
    Real global_mean_sum;
    Real local_l1 = 0;
    Real local_l2 = 0;
    Real local_linf = 0;

    for (int k = 0; k < domain->local[AXIS_Z]; k++) {
        for (int j = 0; j < domain->local[AXIS_Y]; j++) {
            for (int i = 0; i < domain->local[AXIS_X]; i++) {
                size_t index = domain_index(domain, i, j, k);
                local_mean_sum += numerical[index] - exact[index];
            }
        }
    }
    MPI_Allreduce(&local_mean_sum, &global_mean_sum, 1, mpi_real_type(),
                  MPI_SUM, domain->cart);
    Real mean_difference = global_mean_sum / (Real)GRID_CELLS;

    for (int k = 0; k < domain->local[AXIS_Z]; k++) {
        for (int j = 0; j < domain->local[AXIS_Y]; j++) {
            for (int i = 0; i < domain->local[AXIS_X]; i++) {
                size_t index = domain_index(domain, i, j, k);
                Real difference = (Real)fabs(
                    (double)(numerical[index] - exact[index] -
                             mean_difference));

                local_l1 += difference;
                local_l2 += difference * difference;
                if (difference > local_linf) local_linf = difference;
            }
        }
    }
    return reduce_error_norms(local_l1, local_l2, local_linf, domain);
}

static inline void fill_exact_velocity(VectorField *exact,
                                       const Domain *domain,
                                       VectorFunction velocity_fn,
                                       Real time)
{
    for (int k = 0; k < domain->local[AXIS_Z]; k++) {
        int global_k = domain_global_index(domain, k, AXIS_Z);
        for (int j = 0; j < domain->local[AXIS_Y]; j++) {
            int global_j = domain_global_index(domain, j, AXIS_Y);
            for (int i = 0; i < domain->local[AXIS_X]; i++) {
                int global_i = domain_global_index(domain, i, AXIS_X);
                size_t index = domain_index(domain, i, j, k);
                Real x = centered_physical_coord(global_i, AXIS_X);
                Real y = centered_physical_coord(global_j, AXIS_Y);
                Real z = centered_physical_coord(global_k, AXIS_Z);

                exact->v_x[index] = velocity_fn(
                    staggered_physical_coord(global_i, AXIS_X),
                    y, z, time, 0);
                exact->v_y[index] = velocity_fn(
                    x, staggered_physical_coord(global_j, AXIS_Y),
                    z, time, 1);
                exact->v_z[index] = velocity_fn(
                    x, y, staggered_physical_coord(global_k, AXIS_Z),
                    time, 2);
            }
        }
    }
}

static inline void fill_exact_pressure(ScalarField *exact,
                                       const Domain *domain,
                                       ScalarFunction pressure_fn,
                                       Real time)
{
    for (int k = 0; k < domain->local[AXIS_Z]; k++) {
        int global_k = domain_global_index(domain, k, AXIS_Z);
        for (int j = 0; j < domain->local[AXIS_Y]; j++) {
            int global_j = domain_global_index(domain, j, AXIS_Y);
            for (int i = 0; i < domain->local[AXIS_X]; i++) {
                int global_i = domain_global_index(domain, i, AXIS_X);
                size_t index = domain_index(domain, i, j, k);

                exact->v[index] = pressure_fn(
                    centered_physical_coord(global_i, AXIS_X),
                    centered_physical_coord(global_j, AXIS_Y),
                    centered_physical_coord(global_k, AXIS_Z), time);
            }
        }
    }
}

static inline SolverErrorNorms compute_solver_error_norms(
    const SolverMemState *state, const Data *data,
    Real velocity_time, Real pressure_time)
{
    const Domain *domain = &state->domain;
    VectorField exact_velocity;
    ScalarField exact_pressure;

    vectorField_alloc(&exact_velocity, domain);
    scalarField_alloc(&exact_pressure, domain);
    fill_exact_velocity(&exact_velocity, domain, data->velocity_fn,
                        velocity_time);
    fill_exact_pressure(&exact_pressure, domain, data->pressure_fn,
                        pressure_time);

    SolverErrorNorms errors = {
        .velocity_x = compute_error_norms(state->u.v_x,
                                          exact_velocity.v_x, domain),
        .velocity_y = compute_error_norms(state->u.v_y,
                                          exact_velocity.v_y, domain),
        .velocity_z = compute_error_norms(state->u.v_z,
                                          exact_velocity.v_z, domain),
        .pressure = compute_pressure_error_norms(state->pressure.v,
                                                  exact_pressure.v, domain),
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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return;

    printf("\nManufactured-solution error report:\n");
    printf("  Grid: %d x %d x %d\n", WIDTH, HEIGHT, DEPTH);
    printf("  dt: %.2e\n", (double)DT);
    printf("  T:  %.2e\n", (double)T);
    printf("  Velocity verification time: %.2e\n", (double)velocity_time);
    printf("  Pressure verification time: %.2e\n", (double)pressure_time);
    printf("  L2 error u_x: %.4e\n", (double)errors->velocity_x.L2);
    printf("  L2 error u_y: %.4e\n", (double)errors->velocity_y.L2);
    printf("  L2 error u_z: %.4e\n", (double)errors->velocity_z.L2);
    printf("  L2 error p:   %.4e\n", (double)errors->pressure.L2);
}

#endif
