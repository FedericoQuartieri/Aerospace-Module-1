#include "manufactured_cases.h"
#include "kernels.h"
#include "solver_internal.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static void fill_scalar(ScalarField *field, Real phase)
{
    size_t q;
    for (q = 0; q < field->count; ++q) {
        field->data[q] =
            sin((Real)0.013 * (Real)(q + 1) + phase) +
            (Real)0.2 * cos((Real)0.007 * (Real)(q + 3) - phase);
    }
}

static bool fields_match(const ScalarField *standard,
                         const ScalarField *optimized,
                         const char *stage,
                         size_t extent,
                         Direction component)
{
    const Real absolute_tolerance = sizeof(Real) == sizeof(float)
        ? (Real)4e-5 : (Real)4e-12;
    const Real relative_tolerance = sizeof(Real) == sizeof(float)
        ? (Real)4e-5 : (Real)4e-12;
    size_t q;

    for (q = 0; q < standard->count; ++q) {
        const Real scale = fmax(fabs(standard->data[q]),
                                fabs(optimized->data[q]));
        const Real difference = fabs(standard->data[q] -
                                     optimized->data[q]);
        if (!isfinite(standard->data[q]) ||
            !isfinite(optimized->data[q]) ||
            difference > absolute_tolerance + relative_tolerance * scale) {
            fprintf(stderr,
                    "%s mismatch at extent %zu component %d index %zu: "
                    "standard %.17g optimized %.17g\n",
                    stage, extent, (int)component, q,
                    (double)standard->data[q],
                    (double)optimized->data[q]);
            return false;
        }
    }
    return true;
}

static int run_extent(size_t extent)
{
    const ManufacturedCase *test_case = &MANUFACTURED_CASES[0];
    SolverConfig config = test_case->base_config;
    Grid grid;
    ScalarField eta = {0};
    ScalarField zeta = {0};
    ScalarField velocity = {0};
    ScalarField pressure_star = {0};
    ScalarField gamma = {0};
    ScalarField standard_stage = {0};
    ScalarField optimized_stage = {0};
    ScalarField standard_rhs = {0};
    ScalarField optimized_rhs = {0};
    ScalarField pressure_input = {0};
    ScalarField standard_output = {0};
    ScalarField optimized_output = {0};
    VectorField pressure_velocity = {0};
    RealBuffer standard_scratch = {0};
    RealBuffer optimized_scratch = {0};
    Direction component;
    size_t q;
    int result = 1;

    config.extent[DIRECTION_X] = extent;
    config.extent[DIRECTION_Y] = extent;
    config.extent[DIRECTION_Z] = extent;
    config.output_frequency = 0;
    config.output_directory = NULL;
    if (!grid_init(&grid, &config)) goto cleanup;

#define INIT_FIELD(name) \
    do { if (!scalar_field_init(&(name), grid.cell_count)) goto cleanup; } while (0)
    INIT_FIELD(eta);
    INIT_FIELD(zeta);
    INIT_FIELD(velocity);
    INIT_FIELD(pressure_star);
    INIT_FIELD(gamma);
    INIT_FIELD(standard_stage);
    INIT_FIELD(optimized_stage);
    INIT_FIELD(standard_rhs);
    INIT_FIELD(optimized_rhs);
    INIT_FIELD(pressure_input);
    INIT_FIELD(standard_output);
    INIT_FIELD(optimized_output);
#undef INIT_FIELD
    if (!vector_field_init(&pressure_velocity, grid.cell_count) ||
        !real_buffer_init(&standard_scratch,
                          standard_scratch_capacity(&grid)) ||
        !real_buffer_init(&optimized_scratch,
                          optimized_scratch_capacity(&grid))) {
        goto cleanup;
    }

    fill_scalar(&eta, (Real)0.1);
    fill_scalar(&zeta, (Real)0.4);
    fill_scalar(&velocity, (Real)0.7);
    fill_scalar(&pressure_star, (Real)1.0);
    fill_scalar(&pressure_input, (Real)1.3);
    for (component = DIRECTION_X;
         component < DIRECTION_COUNT;
         component = (Direction)(component + 1)) {
        fill_scalar(&pressure_velocity.component[component],
                    (Real)component * (Real)0.31);
    }
    for (q = 0; q < grid.cell_count; ++q) {
        const Real permeability =
            (Real)1 + (Real)0.1 * sin((Real)q * (Real)0.01);
        const Real beta = (Real)1 +
            config.dt * config.viscosity / ((Real)2 * permeability);
        gamma.data[q] = config.dt * config.viscosity /
                        ((Real)2 * beta);
    }

    for (component = DIRECTION_X;
         component < DIRECTION_COUNT;
         component = (Direction)(component + 1)) {
        scalar_field_copy(&standard_stage, &eta);
        scalar_field_copy(&optimized_stage, &eta);
        standard_momentum_solve_x(
            &grid, &config, &test_case->problem,
            &standard_stage, &zeta, &velocity, &pressure_star,
            &gamma, component, 1, &standard_scratch);
        optimized_momentum_solve_x(
            &grid, &config, &test_case->problem,
            &optimized_stage, &zeta, &velocity, &pressure_star,
            &gamma, component, 1, &optimized_scratch);
        if (!fields_match(&standard_stage, &optimized_stage,
                          "momentum-x", extent, component)) goto cleanup;

        scalar_field_copy(&standard_stage, &zeta);
        scalar_field_copy(&optimized_stage, &zeta);
        standard_momentum_solve_y(
            &grid, &config, &test_case->problem,
            &eta, &standard_stage, &standard_rhs, &gamma,
            component, 1, &standard_scratch);
        optimized_momentum_solve_y(
            &grid, &config, &test_case->problem,
            &eta, &optimized_stage, &optimized_rhs, &gamma,
            component, 1, &optimized_scratch);
        if (!fields_match(&standard_stage, &optimized_stage,
                          "momentum-y", extent, component)) goto cleanup;

        scalar_field_copy(&standard_stage, &velocity);
        scalar_field_copy(&optimized_stage, &velocity);
        standard_momentum_solve_z(
            &grid, &config, &test_case->problem,
            &zeta, &standard_stage, &standard_rhs, &gamma,
            component, 1, &standard_scratch);
        optimized_momentum_solve_z(
            &grid, &config, &test_case->problem,
            &zeta, &optimized_stage, &optimized_rhs, &gamma,
            component, 1, &optimized_scratch);
        if (!fields_match(&standard_stage, &optimized_stage,
                          "momentum-z", extent, component)) goto cleanup;
    }

    standard_pressure_solve_x(&grid, &config, &pressure_velocity,
                              &standard_rhs, &standard_output,
                              &standard_scratch);
    optimized_pressure_solve_x(&grid, &config, &pressure_velocity,
                               &optimized_rhs, &optimized_output,
                               &optimized_scratch);
    if (!fields_match(&standard_output, &optimized_output,
                      "pressure-x", extent, DIRECTION_X)) goto cleanup;

    standard_pressure_solve_y(&grid, &pressure_input, &standard_output,
                              &standard_scratch);
    optimized_pressure_solve_y(&grid, &pressure_input, &optimized_output,
                               &optimized_scratch);
    if (!fields_match(&standard_output, &optimized_output,
                      "pressure-y", extent, DIRECTION_Y)) goto cleanup;

    standard_pressure_solve_z(&grid, &pressure_input, &standard_output,
                              &standard_scratch);
    optimized_pressure_solve_z(&grid, &pressure_input, &optimized_output,
                               &optimized_scratch);
    if (!fields_match(&standard_output, &optimized_output,
                      "pressure-z", extent, DIRECTION_Z)) goto cleanup;

    result = 0;

cleanup:
    real_buffer_destroy(&optimized_scratch);
    real_buffer_destroy(&standard_scratch);
    vector_field_destroy(&pressure_velocity);
    scalar_field_destroy(&optimized_output);
    scalar_field_destroy(&standard_output);
    scalar_field_destroy(&pressure_input);
    scalar_field_destroy(&optimized_rhs);
    scalar_field_destroy(&standard_rhs);
    scalar_field_destroy(&optimized_stage);
    scalar_field_destroy(&standard_stage);
    scalar_field_destroy(&gamma);
    scalar_field_destroy(&pressure_star);
    scalar_field_destroy(&velocity);
    scalar_field_destroy(&zeta);
    scalar_field_destroy(&eta);
    return result;
}

int main(void)
{
    return run_extent(17) || run_extent(31);
}
