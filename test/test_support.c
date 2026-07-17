#include "test_support.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static bool scalar_is_finite(const ScalarField *field)
{
    size_t q;
    for (q = 0; q < field->count; ++q) {
        if (!isfinite(field->data[q])) return false;
    }
    return true;
}

static bool state_is_finite(const Solver *solver)
{
    Direction component;
    for (component = DIRECTION_X;
         component < DIRECTION_COUNT;
         component = (Direction)(component + 1)) {
        if (!scalar_is_finite(&solver->state.eta.component[component]) ||
            !scalar_is_finite(&solver->state.zeta.component[component]) ||
            !scalar_is_finite(&solver->state.velocity.component[component])) {
            return false;
        }
    }
    return scalar_is_finite(&solver->state.pressure) &&
           scalar_is_finite(&solver->state.pressure_star) &&
           scalar_is_finite(&solver->gamma);
}

static bool initialized_at_baseline_levels(const Solver *solver,
                                           const ManufacturedCase *test_case)
{
    const Real tolerance = sizeof(Real) == sizeof(float)
        ? (Real)2e-6 : (Real)2e-14;
    size_t i;
    size_t j;
    size_t k;

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
                const Real exact_pressure =
                    test_case->exact_pressure(px, py, pz, (Real)0);
                Direction component;
                if (fabs(solver->state.pressure.data[index] -
                         exact_pressure) > tolerance ||
                    fabs(solver->state.pressure_star.data[index] -
                         exact_pressure) > tolerance) {
                    return false;
                }
                for (component = DIRECTION_X;
                     component < DIRECTION_COUNT;
                     component = (Direction)(component + 1)) {
                    const Real x = grid_velocity_coordinate(
                        &solver->grid, DIRECTION_X, component, i);
                    const Real y = grid_velocity_coordinate(
                        &solver->grid, DIRECTION_Y, component, j);
                    const Real z = grid_velocity_coordinate(
                        &solver->grid, DIRECTION_Z, component, k);
                    const Real exact = test_case->exact_velocity(
                        x, y, z, (Real)0, component);
                    if (fabs(solver->state.eta.component[component]
                                 .data[index] - exact) > tolerance ||
                        fabs(solver->state.zeta.component[component]
                                 .data[index] - exact) > tolerance ||
                        fabs(solver->state.velocity.component[component]
                                 .data[index] - exact) > tolerance) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

static bool memory_model_is_valid(const Solver *solver)
{
    const ScalarField *scalar_fields[] = {
        &solver->gamma,
        &solver->workspace.field,
        &solver->state.pressure,
        &solver->state.pressure_star
    };
    size_t full_grid_values = solver->gamma.count +
                              solver->workspace.field.count +
                              solver->state.pressure.count +
                              solver->state.pressure_star.count;
    Direction component;
    size_t scalar_index;

    for (scalar_index = 0;
         scalar_index < sizeof(scalar_fields) / sizeof(scalar_fields[0]);
         ++scalar_index) {
        if ((uintptr_t)scalar_fields[scalar_index]->data % 64 != 0) {
            return false;
        }
    }
    for (component = DIRECTION_X;
         component < DIRECTION_COUNT;
         component = (Direction)(component + 1)) {
        const ScalarField *fields[] = {
            &solver->state.eta.component[component],
            &solver->state.zeta.component[component],
            &solver->state.velocity.component[component]
        };
        size_t field_index;
        for (field_index = 0; field_index < 3; ++field_index) {
            if ((uintptr_t)fields[field_index]->data % 64 != 0) return false;
            full_grid_values += fields[field_index]->count;
        }
    }
    return full_grid_values == 13 * solver->grid.cell_count;
}

static void compute_velocity_and_boundary_error(
    const Solver *solver,
    const ManufacturedCase *test_case,
    Real final_time,
    ErrorReport *report)
{
    const Real volume = solver->grid.spacing[DIRECTION_X] *
                        solver->grid.spacing[DIRECTION_Y] *
                        solver->grid.spacing[DIRECTION_Z];
    size_t i;
    size_t j;
    size_t k;

    for (k = 0; k < solver->grid.extent[DIRECTION_Z]; ++k) {
        for (j = 0; j < solver->grid.extent[DIRECTION_Y]; ++j) {
            for (i = 0; i < solver->grid.extent[DIRECTION_X]; ++i) {
                const size_t index = grid_index(&solver->grid, i, j, k);
                Direction component;
                for (component = DIRECTION_X;
                     component < DIRECTION_COUNT;
                     component = (Direction)(component + 1)) {
                    const Real x = grid_velocity_coordinate(
                        &solver->grid, DIRECTION_X, component, i);
                    const Real y = grid_velocity_coordinate(
                        &solver->grid, DIRECTION_Y, component, j);
                    const Real z = grid_velocity_coordinate(
                        &solver->grid, DIRECTION_Z, component, k);
                    const Real exact = test_case->exact_velocity(
                        x, y, z, final_time, component);
                    const Real difference = fabs(
                        solver->state.velocity.component[component]
                            .data[index] - exact);
                    report->velocity[component].l2 += difference * difference;
                    if (difference > report->velocity[component].linf) {
                        report->velocity[component].linf = difference;
                    }
                    if ((i == 0 ||
                         i + 1 == solver->grid.extent[DIRECTION_X] ||
                         j == 0 ||
                         j + 1 == solver->grid.extent[DIRECTION_Y] ||
                         k == 0 ||
                         k + 1 == solver->grid.extent[DIRECTION_Z]) &&
                        difference > report->boundary_linf) {
                        report->boundary_linf = difference;
                    }
                }
            }
        }
    }
    for (Direction component = DIRECTION_X;
         component < DIRECTION_COUNT;
         component = (Direction)(component + 1)) {
        report->velocity[component].l2 =
            sqrt(report->velocity[component].l2 * volume);
    }
}

static void compute_pressure_error(const Solver *solver,
                                   const ManufacturedCase *test_case,
                                   Real pressure_time,
                                   ErrorReport *report)
{
    const Real volume = solver->grid.spacing[DIRECTION_X] *
                        solver->grid.spacing[DIRECTION_Y] *
                        solver->grid.spacing[DIRECTION_Z];
    Real mean_offset = (Real)0;
    size_t i;
    size_t j;
    size_t k;

    /* Homogeneous Neumann pressure is defined up to a constant.  Compare only
     * after removing the numerical-to-exact mean offset. */
    for (k = 0; k < solver->grid.extent[DIRECTION_Z]; ++k) {
        for (j = 0; j < solver->grid.extent[DIRECTION_Y]; ++j) {
            for (i = 0; i < solver->grid.extent[DIRECTION_X]; ++i) {
                const size_t index = grid_index(&solver->grid, i, j, k);
                mean_offset += solver->state.pressure.data[index] -
                    test_case->exact_pressure(
                        grid_pressure_coordinate(&solver->grid, DIRECTION_X, i),
                        grid_pressure_coordinate(&solver->grid, DIRECTION_Y, j),
                        grid_pressure_coordinate(&solver->grid, DIRECTION_Z, k),
                        pressure_time);
            }
        }
    }
    mean_offset /= (Real)solver->grid.cell_count;
    for (k = 0; k < solver->grid.extent[DIRECTION_Z]; ++k) {
        for (j = 0; j < solver->grid.extent[DIRECTION_Y]; ++j) {
            for (i = 0; i < solver->grid.extent[DIRECTION_X]; ++i) {
                const size_t index = grid_index(&solver->grid, i, j, k);
                const Real difference = fabs(
                    solver->state.pressure.data[index] -
                    test_case->exact_pressure(
                        grid_pressure_coordinate(&solver->grid, DIRECTION_X, i),
                        grid_pressure_coordinate(&solver->grid, DIRECTION_Y, j),
                        grid_pressure_coordinate(&solver->grid, DIRECTION_Z, k),
                        pressure_time) - mean_offset);
                report->pressure.l2 += difference * difference;
                if (difference > report->pressure.linf) {
                    report->pressure.linf = difference;
                }
            }
        }
    }
    report->pressure.l2 = sqrt(report->pressure.l2 * volume);
}

static void compute_divergence(const Solver *solver, ErrorReport *report)
{
    const Real volume = solver->grid.spacing[DIRECTION_X] *
                        solver->grid.spacing[DIRECTION_Y] *
                        solver->grid.spacing[DIRECTION_Z];
    size_t i;
    size_t j;
    size_t k;

    for (k = 1; k < solver->grid.extent[DIRECTION_Z]; ++k) {
        for (j = 1; j < solver->grid.extent[DIRECTION_Y]; ++j) {
            for (i = 1; i < solver->grid.extent[DIRECTION_X]; ++i) {
                const size_t index = grid_index(&solver->grid, i, j, k);
                const Real divergence =
                    (solver->state.velocity.component[DIRECTION_X]
                         .data[index] -
                     solver->state.velocity.component[DIRECTION_X]
                         .data[index - solver->grid.stride[DIRECTION_X]]) *
                        solver->grid.inverse_spacing[DIRECTION_X] +
                    (solver->state.velocity.component[DIRECTION_Y]
                         .data[index] -
                     solver->state.velocity.component[DIRECTION_Y]
                         .data[index - solver->grid.stride[DIRECTION_Y]]) *
                        solver->grid.inverse_spacing[DIRECTION_Y] +
                    (solver->state.velocity.component[DIRECTION_Z]
                         .data[index] -
                     solver->state.velocity.component[DIRECTION_Z]
                         .data[index - solver->grid.stride[DIRECTION_Z]]) *
                        solver->grid.inverse_spacing[DIRECTION_Z];
                report->divergence_l2 += divergence * divergence;
            }
        }
    }
    report->divergence_l2 = sqrt(report->divergence_l2 * volume);
}

bool run_manufactured_case(const ManufacturedCase *test_case,
                           const SolverConfig *config,
                           ErrorReport *report)
{
    Solver solver = {0};
    SolverConfig run_config = *config;
    SolverStatus status;
    bool success = false;
    Real final_time;

    memset(report, 0, sizeof(*report));
    /* Numerical tests never create solution snapshots; writer behavior has a
     * separate test_output executable. */
    run_config.output_frequency = 0;
    run_config.output_directory = NULL;
    status = solver_init(&solver, &run_config, &test_case->problem);
    if (status != SOLVER_SUCCESS ||
        !initialized_at_baseline_levels(&solver, test_case) ||
        !memory_model_is_valid(&solver)) {
        goto cleanup;
    }
    status = solver_solve(&solver);
    if (status != SOLVER_SUCCESS ||
        solver.stats.completed_steps != run_config.steps ||
        !state_is_finite(&solver)) {
        goto cleanup;
    }

    final_time = (Real)run_config.steps * run_config.dt;
    compute_velocity_and_boundary_error(&solver, test_case, final_time, report);
    compute_pressure_error(&solver, test_case,
                           final_time - run_config.dt / (Real)2, report);
    compute_divergence(&solver, report);
    success = isfinite(combined_velocity_l2(report)) &&
              isfinite(report->pressure.l2) &&
              isfinite(report->pressure.linf) &&
              isfinite(report->divergence_l2) &&
              isfinite(report->boundary_linf);

cleanup:
    solver_destroy(&solver);
    solver_destroy(&solver);
    return success;
}

Real combined_velocity_l2(const ErrorReport *report)
{
    Real sum = (Real)0;
    Direction component;
    for (component = DIRECTION_X;
         component < DIRECTION_COUNT;
         component = (Direction)(component + 1)) {
        sum += report->velocity[component].l2 *
               report->velocity[component].l2;
    }
    return sqrt(sum);
}

Real convergence_order(Real coarse_error,
                       Real fine_error,
                       Real coarse_scale,
                       Real fine_scale)
{
    if (!(coarse_error > (Real)0) || !(fine_error > (Real)0) ||
        !(coarse_scale > fine_scale)) {
        return (Real)-1;
    }
    return log(coarse_error / fine_error) /
           log(coarse_scale / fine_scale);
}
