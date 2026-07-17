/* DESIGN: scalar kernels without dynamic memory.  Each stage solves an increment and
 * immediately adds it to its persistent field.  X remains the unit-stride loop. */
#include "kernels.h"
#include "solver_internal.h"

#include <assert.h>

static size_t maximum_size(size_t a, size_t b)
{
    return a > b ? a : b;
}

size_t standard_scratch_capacity(const Grid *grid)
{
    const size_t nx = grid->extent[DIRECTION_X];
    const size_t longest = maximum_size(grid->extent[DIRECTION_Y],
                                        grid->extent[DIRECTION_Z]);
    const size_t directional = 2 * nx * longest;
    return maximum_size(3 * nx, directional);
}

static void pressure_thomas_line(Real w,
                                 size_t length,
                                 Real *tmp,
                                 Real *rhs,
                                 Real *solution)
{
    size_t index;
    Real inverse_diagonal = (Real)1 / ((Real)1 - (Real)2 * w);

    /* Homogeneous Neumann rows: superdiagonal 2w on the left and diagonal
     * 1-w on the right.  These are intentionally asymmetric. */
    tmp[0] = ((Real)2 * w) * inverse_diagonal;
    rhs[0] *= inverse_diagonal;
    for (index = 1; index + 1 < length; ++index) {
        inverse_diagonal =
            (Real)1 /
            (((Real)1 - (Real)2 * w) - w * tmp[index - 1]);
        tmp[index] = w * inverse_diagonal;
        rhs[index] = (rhs[index] - w * rhs[index - 1]) *
                     inverse_diagonal;
    }

    inverse_diagonal =
        (Real)1 / (((Real)1 - w) - w * tmp[length - 2]);
    rhs[length - 1] =
        (rhs[length - 1] - w * rhs[length - 2]) * inverse_diagonal;
    solution[length - 1] = rhs[length - 1];

    index = length - 1;
    while (index-- > 0) {
        solution[index] = rhs[index] - tmp[index] * solution[index + 1];
    }
}

void standard_momentum_solve_x(const Grid *grid,
                               const SolverConfig *config,
                               const ProblemDefinition *problem,
                               ScalarField *eta,
                               const ScalarField *zeta,
                               const ScalarField *velocity,
                               const ScalarField *pressure_star,
                               const ScalarField *gamma,
                               Direction component,
                               size_t timestep,
                               RealBuffer *scratch)
{
    const size_t nx = grid->extent[DIRECTION_X];
    const size_t ny = grid->extent[DIRECTION_Y];
    const size_t nz = grid->extent[DIRECTION_Z];
    Real *tmp = scratch->data;
    Real *rhs = tmp + nx;
    Real *increment = rhs + nx;
    size_t j;
    size_t k;

    assert(scratch->capacity >= 3 * nx);
    for (k = 0; k < nz; ++k) {
        for (j = 0; j < ny; ++j) {
            const size_t offset = grid_index(grid, 0, j, k);
            Real inverse_diagonal;
            size_t i;

            tmp[0] = (Real)0;
            rhs[0] = evaluate_velocity_boundary_increment(
                grid, config, problem, 0, j, k, timestep, component);

            for (i = 1; i + 1 < nx; ++i) {
                const size_t index = offset + i;
                const Real w = -gamma->data[index] *
                               grid->inverse_spacing_square[DIRECTION_X];
                inverse_diagonal =
                    (Real)1 /
                    (((Real)1 - (Real)2 * w) - w * tmp[i - 1]);
                tmp[i] = w * inverse_diagonal;
                rhs[i] = evaluate_momentum_x_rhs(
                    grid, config, problem, eta, zeta, velocity,
                    pressure_star, gamma, i, j, k, timestep, component);
                rhs[i] = (rhs[i] - w * rhs[i - 1]) * inverse_diagonal;
            }

            if (component == DIRECTION_X) {
                increment[nx - 1] =
                    evaluate_velocity_boundary_increment(
                        grid, config, problem, nx - 1, j, k,
                        timestep, component);
            } else {
                const size_t index = offset + nx - 1;
                const Real w = -gamma->data[index] *
                               grid->inverse_spacing_square[DIRECTION_X];
                const Real right_boundary =
                    evaluate_velocity_boundary_increment(
                        grid, config, problem, nx - 1, j, k,
                        timestep, component);
                rhs[nx - 1] = evaluate_momentum_x_rhs(
                    grid, config, problem, eta, zeta, velocity,
                    pressure_star, gamma, nx - 1, j, k, timestep,
                    component);
                rhs[nx - 1] -= (Real)2 * w * right_boundary;
                inverse_diagonal =
                    (Real)1 /
                    (((Real)1 - (Real)3 * w) - w * tmp[nx - 2]);
                rhs[nx - 1] =
                    (rhs[nx - 1] - w * rhs[nx - 2]) * inverse_diagonal;
                increment[nx - 1] = rhs[nx - 1];
            }

            i = nx - 1;
            while (i-- > 0) {
                increment[i] = rhs[i] - tmp[i] * increment[i + 1];
            }
            for (i = 0; i < nx; ++i) {
                eta->data[offset + i] += increment[i];
            }
        }
    }
}

static void standard_momentum_directional(
    const Grid *grid,
    const SolverConfig *config,
    const ProblemDefinition *problem,
    const ScalarField *source,
    ScalarField *stage,
    ScalarField *rhs_workspace,
    const ScalarField *gamma,
    Direction component,
    size_t timestep,
    RealBuffer *scratch,
    Direction solve_direction)
{
    const size_t nx = grid->extent[DIRECTION_X];
    const size_t line_length = grid->extent[solve_direction];
    const size_t plane_size = nx * line_length;
    const size_t stride = grid->stride[solve_direction];
    const size_t outer_count = solve_direction == DIRECTION_Y
        ? grid->extent[DIRECTION_Z]
        : grid->extent[DIRECTION_Y];
    Real *tmp = scratch->data;
    Real *increment = tmp + plane_size;
    size_t q;
    size_t outer;

    assert(scratch->capacity >= 2 * plane_size);
    for (q = 0; q < grid->cell_count; ++q) {
        rhs_workspace->data[q] = source->data[q] - stage->data[q];
    }

    for (outer = 0; outer < outer_count; ++outer) {
        const size_t offset = solve_direction == DIRECTION_Y
            ? grid_index(grid, 0, 0, outer)
            : grid_index(grid, 0, outer, 0);
        size_t i;
        size_t level;

        for (i = 0; i < nx; ++i) {
            rhs_workspace->data[offset + i] =
                evaluate_velocity_boundary_increment(
                    grid, config, problem,
                    i,
                    solve_direction == DIRECTION_Y ? 0 : outer,
                    solve_direction == DIRECTION_Z ? 0 : outer,
                    timestep, component);
            tmp[i] = (Real)0;
        }

        for (level = 1; level + 1 < line_length; ++level) {
            for (i = 0; i < nx; ++i) {
                const size_t index = offset + level * stride + i;
                const size_t local = level * nx + i;
                const Real w = -gamma->data[index] *
                    grid->inverse_spacing_square[solve_direction];
                const Real inverse_diagonal =
                    (Real)1 /
                    (((Real)1 - (Real)2 * w) -
                     w * tmp[local - nx]);
                tmp[local] = w * inverse_diagonal;
                rhs_workspace->data[index] =
                    (rhs_workspace->data[index] -
                     w * rhs_workspace->data[index - stride]) *
                    inverse_diagonal;
            }
        }

        for (i = 0; i < nx; ++i) {
            const size_t index = offset + (line_length - 1) * stride + i;
            const size_t local = (line_length - 1) * nx + i;
            const size_t j = solve_direction == DIRECTION_Y
                ? line_length - 1 : outer;
            const size_t k = solve_direction == DIRECTION_Z
                ? line_length - 1 : outer;
            const Real right_boundary =
                evaluate_velocity_boundary_increment(
                    grid, config, problem, i, j, k, timestep, component);

            if (component == solve_direction) {
                increment[local] = right_boundary;
            } else {
                const Real w = -gamma->data[index] *
                    grid->inverse_spacing_square[solve_direction];
                const Real inverse_diagonal =
                    (Real)1 /
                    (((Real)1 - (Real)3 * w) -
                     w * tmp[local - nx]);
                rhs_workspace->data[index] -=
                    (Real)2 * w * right_boundary;
                rhs_workspace->data[index] =
                    (rhs_workspace->data[index] -
                     w * rhs_workspace->data[index - stride]) *
                    inverse_diagonal;
                increment[local] = rhs_workspace->data[index];
            }
        }

        level = line_length - 1;
        while (level-- > 0) {
            for (i = 0; i < nx; ++i) {
                const size_t index = offset + level * stride + i;
                const size_t local = level * nx + i;
                increment[local] = rhs_workspace->data[index] -
                                   tmp[local] * increment[local + nx];
            }
        }

        for (level = 0; level < line_length; ++level) {
            for (i = 0; i < nx; ++i) {
                stage->data[offset + level * stride + i] +=
                    increment[level * nx + i];
            }
        }
    }
}

void standard_momentum_solve_y(const Grid *grid,
                               const SolverConfig *config,
                               const ProblemDefinition *problem,
                               const ScalarField *source,
                               ScalarField *stage,
                               ScalarField *rhs_workspace,
                               const ScalarField *gamma,
                               Direction component,
                               size_t timestep,
                               RealBuffer *scratch)
{
    standard_momentum_directional(grid, config, problem, source, stage,
                                  rhs_workspace, gamma, component, timestep,
                                  scratch, DIRECTION_Y);
}

void standard_momentum_solve_z(const Grid *grid,
                               const SolverConfig *config,
                               const ProblemDefinition *problem,
                               const ScalarField *source,
                               ScalarField *stage,
                               ScalarField *rhs_workspace,
                               const ScalarField *gamma,
                               Direction component,
                               size_t timestep,
                               RealBuffer *scratch)
{
    standard_momentum_directional(grid, config, problem, source, stage,
                                  rhs_workspace, gamma, component, timestep,
                                  scratch, DIRECTION_Z);
}

void standard_pressure_solve_x(const Grid *grid,
                               const SolverConfig *config,
                               const VectorField *velocity,
                               ScalarField *rhs_workspace,
                               ScalarField *psi,
                               RealBuffer *scratch)
{
    const size_t nx = grid->extent[DIRECTION_X];
    const size_t ny = grid->extent[DIRECTION_Y];
    const size_t nz = grid->extent[DIRECTION_Z];
    const Real w = -grid->inverse_spacing_square[DIRECTION_X];
    Real *tmp = scratch->data;
    size_t i;
    size_t j;
    size_t k;

    assert(scratch->capacity >= nx);
    for (k = 0; k < nz; ++k) {
        for (j = 0; j < ny; ++j) {
            for (i = 0; i < nx; ++i) {
                const size_t index = grid_index(grid, i, j, k);
                if (i == 0 || j == 0 || k == 0) {
                    rhs_workspace->data[index] = (Real)0;
                } else {
                    const Real divergence =
                        (velocity->component[DIRECTION_X].data[index] -
                         velocity->component[DIRECTION_X]
                             .data[index - grid->stride[DIRECTION_X]]) *
                            grid->inverse_spacing[DIRECTION_X] +
                        (velocity->component[DIRECTION_Y].data[index] -
                         velocity->component[DIRECTION_Y]
                             .data[index - grid->stride[DIRECTION_Y]]) *
                            grid->inverse_spacing[DIRECTION_Y] +
                        (velocity->component[DIRECTION_Z].data[index] -
                         velocity->component[DIRECTION_Z]
                             .data[index - grid->stride[DIRECTION_Z]]) *
                            grid->inverse_spacing[DIRECTION_Z];
                    rhs_workspace->data[index] = -divergence / config->dt;
                }
            }
        }
    }

    for (k = 0; k < nz; ++k) {
        for (j = 0; j < ny; ++j) {
            const size_t offset = grid_index(grid, 0, j, k);
            pressure_thomas_line(w, nx, tmp,
                                 rhs_workspace->data + offset,
                                 psi->data + offset);
        }
    }
}

static void standard_pressure_directional(const Grid *grid,
                                          const ScalarField *input,
                                          ScalarField *output,
                                          RealBuffer *scratch,
                                          Direction direction)
{
    const size_t nx = grid->extent[DIRECTION_X];
    const size_t line_length = grid->extent[direction];
    const size_t stride = grid->stride[direction];
    const size_t outer_count = direction == DIRECTION_Y
        ? grid->extent[DIRECTION_Z]
        : grid->extent[DIRECTION_Y];
    const Real w = -grid->inverse_spacing_square[direction];
    Real *tmp = scratch->data;
    size_t outer;

    assert(scratch->capacity >= nx * line_length);
    for (outer = 0; outer < outer_count; ++outer) {
        const size_t offset = direction == DIRECTION_Y
            ? grid_index(grid, 0, 0, outer)
            : grid_index(grid, 0, outer, 0);
        size_t i;
        size_t level;

        for (i = 0; i < nx; ++i) {
            const Real inverse_diagonal =
                (Real)1 / ((Real)1 - (Real)2 * w);
            tmp[i] = ((Real)2 * w) * inverse_diagonal;
            output->data[offset + i] =
                input->data[offset + i] * inverse_diagonal;
        }

        for (level = 1; level + 1 < line_length; ++level) {
            for (i = 0; i < nx; ++i) {
                const size_t index = offset + level * stride + i;
                const size_t local = level * nx + i;
                const Real inverse_diagonal =
                    (Real)1 /
                    (((Real)1 - (Real)2 * w) -
                     w * tmp[local - nx]);
                tmp[local] = w * inverse_diagonal;
                output->data[index] =
                    (input->data[index] - w * output->data[index - stride]) *
                    inverse_diagonal;
            }
        }

        for (i = 0; i < nx; ++i) {
            const size_t index = offset + (line_length - 1) * stride + i;
            const size_t local = (line_length - 1) * nx + i;
            const Real inverse_diagonal =
                (Real)1 / (((Real)1 - w) - w * tmp[local - nx]);
            output->data[index] =
                (input->data[index] - w * output->data[index - stride]) *
                inverse_diagonal;
        }

        level = line_length - 1;
        while (level-- > 0) {
            for (i = 0; i < nx; ++i) {
                const size_t index = offset + level * stride + i;
                output->data[index] -=
                    tmp[level * nx + i] * output->data[index + stride];
            }
        }
    }
}

void standard_pressure_solve_y(const Grid *grid,
                               const ScalarField *input,
                               ScalarField *output,
                               RealBuffer *scratch)
{
    standard_pressure_directional(grid, input, output, scratch, DIRECTION_Y);
}

void standard_pressure_solve_z(const Grid *grid,
                               const ScalarField *input,
                               ScalarField *output,
                               RealBuffer *scratch)
{
    standard_pressure_directional(grid, input, output, scratch, DIRECTION_Z);
}
