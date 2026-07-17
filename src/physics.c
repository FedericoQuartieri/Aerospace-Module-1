#include "solver_internal.h"

static Real boundary_increment_at(const SolverConfig *config,
                                  const ProblemDefinition *problem,
                                  Real x,
                                  Real y,
                                  Real z,
                                  size_t timestep,
                                  Direction component)
{
    const Real time = (Real)timestep * config->dt;
    return evaluate_velocity_boundary(problem, x, y, z, time, component) -
           evaluate_velocity_boundary(problem, x, y, z,
                                      time - config->dt, component);
}

Real evaluate_velocity_boundary(const ProblemDefinition *problem,
                                Real x,
                                Real y,
                                Real z,
                                Real time,
                                Direction component)
{
    return problem->boundary_velocity(x, y, z, time, component);
}

/*
 * The split systems solve velocity increments.  Lower faces have priority over
 * upper faces at edges and corners, matching the baseline's overwrite order.
 * The normal value on a lower face includes the discrete divergence correction.
 */
Real evaluate_velocity_boundary_increment(const Grid *grid,
                                          const SolverConfig *config,
                                          const ProblemDefinition *problem,
                                          size_t i,
                                          size_t j,
                                          size_t k,
                                          size_t timestep,
                                          Direction component)
{
    const size_t nx = grid->extent[DIRECTION_X];
    const size_t ny = grid->extent[DIRECTION_Y];
    const size_t nz = grid->extent[DIRECTION_Z];
    const Real dx = grid->spacing[DIRECTION_X];
    const Real dy = grid->spacing[DIRECTION_Y];
    const Real dz = grid->spacing[DIRECTION_Z];
    const Real x = grid_pressure_coordinate(grid, DIRECTION_X, i);
    const Real y = grid_pressure_coordinate(grid, DIRECTION_Y, j);
    const Real z = grid_pressure_coordinate(grid, DIRECTION_Z, k);
    const Real vx = grid_velocity_coordinate(grid, DIRECTION_X, DIRECTION_X, i);
    const Real vy = grid_velocity_coordinate(grid, DIRECTION_Y, DIRECTION_Y, j);
    const Real vz = grid_velocity_coordinate(grid, DIRECTION_Z, DIRECTION_Z, k);

#define DELTA(px, py, pz, c) \
    boundary_increment_at(config, problem, (px), (py), (pz), timestep, (c))

    if (i == 0 && j == 0 && k == 0) {
        if (component == DIRECTION_X) return DELTA(dx / 2, 0, 0, component);
        if (component == DIRECTION_Y) return DELTA(0, dy / 2, 0, component);
        return DELTA(0, 0, dz / 2, component);
    }
    if (i == 0 && j == 0) {
        if (component == DIRECTION_X) return DELTA(dx / 2, 0, z, component);
        if (component == DIRECTION_Y) return DELTA(0, dy / 2, z, component);
        return DELTA(0, 0, vz, component);
    }
    if (i == 0 && k == 0) {
        if (component == DIRECTION_X) return DELTA(dx / 2, y, 0, component);
        if (component == DIRECTION_Y) return DELTA(0, vy, 0, component);
        return DELTA(0, y, dz / 2, component);
    }
    if (j == 0 && k == 0) {
        if (component == DIRECTION_X) return DELTA(vx, 0, 0, component);
        if (component == DIRECTION_Y) return DELTA(x, dy / 2, 0, component);
        return DELTA(x, 0, dz / 2, component);
    }
    if (i == 0) {
        if (component == DIRECTION_X) {
            const Real transverse_y =
                (DELTA(0, vy, z, DIRECTION_Y) -
                 DELTA(0, vy - dy, z, DIRECTION_Y)) *
                grid->inverse_spacing[DIRECTION_Y];
            const Real transverse_z =
                (DELTA(0, y, vz, DIRECTION_Z) -
                 DELTA(0, y, vz - dz, DIRECTION_Z)) *
                grid->inverse_spacing[DIRECTION_Z];
            return DELTA(0, y, z, component) -
                   dx * (transverse_y + transverse_z) / 2;
        }
        if (component == DIRECTION_Y) return DELTA(0, vy, z, component);
        return DELTA(0, y, vz, component);
    }
    if (j == 0) {
        if (component == DIRECTION_X) return DELTA(vx, 0, z, component);
        if (component == DIRECTION_Y) {
            const Real transverse_x =
                (DELTA(vx, 0, z, DIRECTION_X) -
                 DELTA(vx - dx, 0, z, DIRECTION_X)) *
                grid->inverse_spacing[DIRECTION_X];
            const Real transverse_z =
                (DELTA(x, 0, vz, DIRECTION_Z) -
                 DELTA(x, 0, vz - dz, DIRECTION_Z)) *
                grid->inverse_spacing[DIRECTION_Z];
            return DELTA(x, 0, z, component) -
                   dy * (transverse_x + transverse_z) / 2;
        }
        return DELTA(x, 0, vz, component);
    }
    if (k == 0) {
        if (component == DIRECTION_X) return DELTA(vx, y, 0, component);
        if (component == DIRECTION_Y) return DELTA(x, vy, 0, component);
        {
            const Real transverse_x =
                (DELTA(vx, y, 0, DIRECTION_X) -
                 DELTA(vx - dx, y, 0, DIRECTION_X)) *
                grid->inverse_spacing[DIRECTION_X];
            const Real transverse_y =
                (DELTA(x, vy, 0, DIRECTION_Y) -
                 DELTA(x, vy - dy, 0, DIRECTION_Y)) *
                grid->inverse_spacing[DIRECTION_Y];
            return DELTA(x, y, 0, component) -
                   dz * (transverse_x + transverse_y) / 2;
        }
    }

    if (i == nx - 1 && j == ny - 1 && k == nz - 1) {
        if (component == DIRECTION_X) return DELTA(vx, y, vz, component);
        if (component == DIRECTION_Y) return DELTA(x, vy, vz, component);
        return DELTA(x, y, vz, component);
    }
    if (i == nx - 1 && j == ny - 1) {
        if (component == DIRECTION_X) return DELTA(vx, vy, z, component);
        if (component == DIRECTION_Y) return DELTA(x, vy, z, component);
        return DELTA(x, vy, vz, component);
    }
    if (i == nx - 1 && k == nz - 1) {
        if (component == DIRECTION_X) return DELTA(vx, y, vz, component);
        if (component == DIRECTION_Y) return DELTA(x, vy, vz, component);
        return DELTA(x, y, vz, component);
    }
    if (j == ny - 1 && k == nz - 1) {
        if (component == DIRECTION_X) return DELTA(vx, y, vz, component);
        if (component == DIRECTION_Y) return DELTA(x, vy, vz, component);
        return DELTA(x, y, vz, component);
    }
    if (i == nx - 1) {
        if (component == DIRECTION_X) return DELTA(vx, y, z, component);
        if (component == DIRECTION_Y) return DELTA(vx, vy, z, component);
        return DELTA(vx, y, vz, component);
    }
    if (j == ny - 1) {
        if (component == DIRECTION_X) return DELTA(vx, vy, z, component);
        if (component == DIRECTION_Y) return DELTA(x, vy, z, component);
        return DELTA(x, vy, vz, component);
    }
    if (k == nz - 1) {
        if (component == DIRECTION_X) return DELTA(vx, y, vz, component);
        if (component == DIRECTION_Y) return DELTA(x, vy, vz, component);
        return DELTA(x, y, vz, component);
    }

#undef DELTA
    return (Real)0;
}

static Real interior_second_difference(const Grid *grid,
                                       const ScalarField *field,
                                       size_t index,
                                       Direction direction)
{
    const size_t stride = grid->stride[direction];
    return (field->data[index - stride] -
            (Real)2 * field->data[index] +
            field->data[index + stride]) *
           grid->inverse_spacing_square[direction];
}

static Real upper_second_difference(const Grid *grid,
                                    const ProblemDefinition *problem,
                                    const ScalarField *field,
                                    size_t index,
                                    size_t i,
                                    size_t j,
                                    size_t k,
                                    Real time,
                                    Direction derivative_direction,
                                    Direction component)
{
    Real coordinate[DIRECTION_COUNT];
    const size_t stride = grid->stride[derivative_direction];

    coordinate[DIRECTION_X] =
        grid_velocity_coordinate(grid, DIRECTION_X, component, i);
    coordinate[DIRECTION_Y] =
        grid_velocity_coordinate(grid, DIRECTION_Y, component, j);
    coordinate[DIRECTION_Z] =
        grid_velocity_coordinate(grid, DIRECTION_Z, component, k);
    coordinate[derivative_direction] =
        grid_velocity_coordinate(grid, derivative_direction,
                                 derivative_direction,
                                 grid->extent[derivative_direction] - 1);

    return (field->data[index - stride] -
            (Real)3 * field->data[index] +
            (Real)2 * evaluate_velocity_boundary(problem,
                                                 coordinate[DIRECTION_X],
                                                 coordinate[DIRECTION_Y],
                                                 coordinate[DIRECTION_Z],
                                                 time,
                                                 component)) *
           grid->inverse_spacing_square[derivative_direction];
}

Real evaluate_g(const Grid *grid,
                const SolverConfig *config,
                const ProblemDefinition *problem,
                const ScalarField *eta,
                const ScalarField *zeta,
                const ScalarField *velocity,
                const ScalarField *pressure_star,
                const ScalarField *gamma,
                size_t i,
                size_t j,
                size_t k,
                size_t timestep,
                Direction component)
{
    const size_t nx = grid->extent[DIRECTION_X];
    const size_t ny = grid->extent[DIRECTION_Y];
    const size_t nz = grid->extent[DIRECTION_Z];
    const size_t index = grid_index(grid, i, j, k);
    const Real forcing_time = ((Real)timestep - (Real)0.5) * config->dt;
    const Real velocity_time = ((Real)timestep - (Real)1) * config->dt;
    Real coordinate[DIRECTION_COUNT];
    Real laplacian_x;
    Real laplacian_y;
    Real laplacian_z;
    Real pressure_gradient;
    Real beta;
    Real permeability;

    if (component == DIRECTION_X &&
        !(i >= 1 && i <= nx - 2 && j >= 1 && j <= ny - 1 &&
          k >= 1 && k <= nz - 1)) {
        return (Real)0;
    }
    if (component == DIRECTION_Y &&
        !(i >= 1 && i <= nx - 1 && j >= 1 && j <= ny - 2 &&
          k >= 1 && k <= nz - 1)) {
        return (Real)0;
    }
    if (component == DIRECTION_Z &&
        !(i >= 1 && i <= nx - 1 && j >= 1 && j <= ny - 1 &&
          k >= 1 && k <= nz - 2)) {
        return (Real)0;
    }

    coordinate[DIRECTION_X] =
        grid_velocity_coordinate(grid, DIRECTION_X, component, i);
    coordinate[DIRECTION_Y] =
        grid_velocity_coordinate(grid, DIRECTION_Y, component, j);
    coordinate[DIRECTION_Z] =
        grid_velocity_coordinate(grid, DIRECTION_Z, component, k);

    /* The component velocity is half a cell above the pressure point, so the
     * baseline gradient uses the pressure value in the positive direction. */
    pressure_gradient =
        (pressure_star->data[index + grid->stride[component]] -
         pressure_star->data[index]) *
        grid->inverse_spacing[component];

    laplacian_x = i == nx - 1
        ? upper_second_difference(grid, problem, eta, index, i, j, k,
                                  velocity_time, DIRECTION_X, component)
        : interior_second_difference(grid, eta, index, DIRECTION_X);
    laplacian_y = j == ny - 1
        ? upper_second_difference(grid, problem, zeta, index, i, j, k,
                                  velocity_time, DIRECTION_Y, component)
        : interior_second_difference(grid, zeta, index, DIRECTION_Y);
    laplacian_z = k == nz - 1
        ? upper_second_difference(grid, problem, velocity, index, i, j, k,
                                  velocity_time, DIRECTION_Z, component)
        : interior_second_difference(grid, velocity, index, DIRECTION_Z);

    beta = config->dt * config->viscosity /
           ((Real)2 * gamma->data[index]);
    permeability = config->dt * config->viscosity /
                   ((Real)2 * (beta - (Real)1));

    return problem->forcing(coordinate[DIRECTION_X],
                            coordinate[DIRECTION_Y],
                            coordinate[DIRECTION_Z],
                            forcing_time,
                            component) -
           pressure_gradient -
           (config->viscosity / permeability) * velocity->data[index] +
           config->viscosity *
               (laplacian_x + laplacian_y + laplacian_z);
}

Real evaluate_momentum_x_rhs(const Grid *grid,
                             const SolverConfig *config,
                             const ProblemDefinition *problem,
                             const ScalarField *eta,
                             const ScalarField *zeta,
                             const ScalarField *velocity,
                             const ScalarField *pressure_star,
                             const ScalarField *gamma,
                             size_t i,
                             size_t j,
                             size_t k,
                             size_t timestep,
                             Direction component)
{
    const size_t index = grid_index(grid, i, j, k);
    const Real beta = config->dt * config->viscosity /
                      ((Real)2 * gamma->data[index]);
    const Real xi = velocity->data[index] +
                    (config->dt / beta) *
                        evaluate_g(grid, config, problem, eta, zeta, velocity,
                                   pressure_star, gamma, i, j, k, timestep,
                                   component);
    return xi - eta->data[index];
}
