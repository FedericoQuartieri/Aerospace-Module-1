#include "solver.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define TUNNEL_LENGTH_X ((Real)2)
#define TUNNEL_LENGTH_Y ((Real)1)
#define TUNNEL_LENGTH_Z ((Real)1)
#define INLET_VELOCITY ((Real)3)
#define FLUID_PERMEABILITY ((Real)1)
#define SOLID_PERMEABILITY ((Real)0.002)
#define TUNNEL_OBSTACLE_RADIUS ((Real)0.20)
#define CAVITY_SIDE ((Real)1)
#define CAVITY_EXPANSION_START ((Real)0.5)
#define CAVITY_INLET_HALF_WIDTH ((Real)0.15)
#define CAVITY_OUTLET_HALF_WIDTH ((Real)0.45)

static Real zero_velocity(Real x,
                          Real y,
                          Real z,
                          Real time,
                          Direction component)
{
    (void)x;
    (void)y;
    (void)z;
    (void)time;
    (void)component;
    return (Real)0;
}

static Real zero_pressure(Real x, Real y, Real z, Real time)
{
    (void)x;
    (void)y;
    (void)z;
    (void)time;
    return (Real)0;
}

static Real tunnel_boundary_velocity(Real x,
                                     Real y,
                                     Real z,
                                     Real time,
                                     Direction component)
{
    const bool on_open_x_face =
        x <= (Real)0 || x >= TUNNEL_LENGTH_X;
    const bool away_from_wall_edges =
        y > (Real)0 && y < TUNNEL_LENGTH_Y &&
        z > (Real)0 && z < TUNNEL_LENGTH_Z;

    /* The first step switches on a spatially uniform inlet.  The same value
     * on the opposite face is the outlet required by incompressibility. */
    if (time > (Real)0 && component == DIRECTION_X &&
        on_open_x_face && away_from_wall_edges) {
        return INLET_VELOCITY;
    }
    return (Real)0;
}

static Real cavity_boundary_velocity(Real x,
                                     Real y,
                                     Real z,
                                     Real time,
                                     Direction component)
{
    const Real center = CAVITY_SIDE / (Real)2;
    const bool inside_inlet =
        fabs(y - center) <= CAVITY_INLET_HALF_WIDTH &&
        fabs(z - center) <= CAVITY_INLET_HALF_WIDTH;
    const bool inside_outlet =
        fabs(y - center) <= CAVITY_OUTLET_HALF_WIDTH &&
        fabs(z - center) <= CAVITY_OUTLET_HALF_WIDTH;
    const Real width_ratio =
        CAVITY_INLET_HALF_WIDTH / CAVITY_OUTLET_HALF_WIDTH;

    if (time <= (Real)0 || component != DIRECTION_X) return (Real)0;
    if (x <= (Real)0 && inside_inlet) return INLET_VELOCITY;
    if (x >= CAVITY_SIDE && inside_outlet) {
        return INLET_VELOCITY * width_ratio * width_ratio;
    }
    return (Real)0;
}

static Real tunnel_permeability(Real x, Real y, Real z, Real time)
{
    const Real distance_x = x - TUNNEL_LENGTH_X / (Real)2;
    const Real distance_y = y - TUNNEL_LENGTH_Y / (Real)2;
    const Real distance_z = z - TUNNEL_LENGTH_Z / (Real)2;
    const Real distance_square =
        distance_x * distance_x +
        distance_y * distance_y +
        distance_z * distance_z;

    (void)time;
    return distance_square <=
            TUNNEL_OBSTACLE_RADIUS * TUNNEL_OBSTACLE_RADIUS
        ? SOLID_PERMEABILITY
        : FLUID_PERMEABILITY;
}

static Real cavity_half_width(Real x)
{
    if (x <= CAVITY_EXPANSION_START) {
        return CAVITY_INLET_HALF_WIDTH;
    }
    if (x >= CAVITY_SIDE) return CAVITY_OUTLET_HALF_WIDTH;
    return CAVITY_INLET_HALF_WIDTH +
        (CAVITY_OUTLET_HALF_WIDTH - CAVITY_INLET_HALF_WIDTH) *
        (x - CAVITY_EXPANSION_START) /
        (CAVITY_SIDE - CAVITY_EXPANSION_START);
}

static Real cavity_permeability(Real x, Real y, Real z, Real time)
{
    const Real center = CAVITY_SIDE / (Real)2;
    const Real half_width = cavity_half_width(x);

    (void)time;
    return fabs(y - center) <= half_width &&
           fabs(z - center) <= half_width
        ? FLUID_PERMEABILITY
        : SOLID_PERMEABILITY;
}

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

static bool gamma_contains_fluid_and_solid(const Solver *solver)
{
    const Real relative_tolerance = sizeof(Real) == sizeof(float)
        ? (Real)2e-5 : (Real)2e-13;
    bool found_fluid = false;
    bool found_solid = false;
    size_t i;
    size_t j;
    size_t k;

    for (k = 0; k < solver->grid.extent[DIRECTION_Z]; ++k) {
        for (j = 0; j < solver->grid.extent[DIRECTION_Y]; ++j) {
            for (i = 0; i < solver->grid.extent[DIRECTION_X]; ++i) {
                const size_t index = grid_index(&solver->grid, i, j, k);
                const Real permeability = solver->problem->permeability(
                    grid_pressure_coordinate(
                        &solver->grid, DIRECTION_X, i),
                    grid_pressure_coordinate(
                        &solver->grid, DIRECTION_Y, j),
                    grid_pressure_coordinate(
                        &solver->grid, DIRECTION_Z, k),
                    (Real)0);
                const Real beta = (Real)1 +
                    solver->config.dt * solver->config.viscosity /
                    ((Real)2 * permeability);
                const Real expected_gamma =
                    solver->config.dt * solver->config.viscosity /
                    ((Real)2 * beta);

                if (fabs(solver->gamma.data[index] - expected_gamma) >
                    relative_tolerance * fabs(expected_gamma)) {
                    return false;
                }
                found_fluid = found_fluid ||
                    permeability == FLUID_PERMEABILITY;
                found_solid = found_solid ||
                    permeability == SOLID_PERMEABILITY;
            }
        }
    }
    return found_fluid && found_solid;
}

static bool open_faces_have_prescribed_velocity(const Solver *solver)
{
    const Real tolerance = sizeof(Real) == sizeof(float)
        ? (Real)2e-5 : (Real)2e-12;
    const size_t last_x = solver->grid.extent[DIRECTION_X] - 1;
    const Real final_time =
        (Real)solver->config.steps * solver->config.dt;
    size_t inlet_nodes = 0;
    size_t outlet_nodes = 0;
    size_t j;
    size_t k;

    /* Edge nodes are shared with stationary walls, so only the interior of
     * each open face is part of the constant inlet/outlet assertion. */
    for (k = 1; k + 1 < solver->grid.extent[DIRECTION_Z]; ++k) {
        for (j = 1; j + 1 < solver->grid.extent[DIRECTION_Y]; ++j) {
            const Real y = grid_pressure_coordinate(
                &solver->grid, DIRECTION_Y, j);
            const Real z = grid_pressure_coordinate(
                &solver->grid, DIRECTION_Z, k);
            const Real expected_inlet = solver->problem->boundary_velocity(
                (Real)0, y, z, final_time, DIRECTION_X);
            const Real expected_outlet = solver->problem->boundary_velocity(
                solver->grid.length[DIRECTION_X], y, z,
                final_time, DIRECTION_X);
            const size_t inlet = grid_index(&solver->grid, 0, j, k);
            const size_t outlet = grid_index(
                &solver->grid, last_x, j, k);

            if (fabs(solver->state.velocity.component[DIRECTION_X]
                         .data[inlet] - expected_inlet) > tolerance ||
                fabs(solver->state.velocity.component[DIRECTION_X]
                         .data[outlet] - expected_outlet) > tolerance) {
                return false;
            }
            if (expected_inlet != (Real)0) ++inlet_nodes;
            if (expected_outlet != (Real)0) ++outlet_nodes;
        }
    }
    return inlet_nodes > 0 && outlet_nodes > 0;
}

static bool wall_normal_velocity_is_zero(const Solver *solver)
{
    const Real tolerance = sizeof(Real) == sizeof(float)
        ? (Real)2e-5 : (Real)2e-12;
    const size_t last_y = solver->grid.extent[DIRECTION_Y] - 1;
    const size_t last_z = solver->grid.extent[DIRECTION_Z] - 1;
    size_t i;
    size_t j;
    size_t k;

    for (k = 0; k < solver->grid.extent[DIRECTION_Z]; ++k) {
        for (i = 0; i < solver->grid.extent[DIRECTION_X]; ++i) {
            const size_t lower = grid_index(&solver->grid, i, 0, k);
            const size_t upper = grid_index(&solver->grid, i, last_y, k);

            if (fabs(solver->state.velocity.component[DIRECTION_Y]
                         .data[lower]) > tolerance ||
                fabs(solver->state.velocity.component[DIRECTION_Y]
                         .data[upper]) > tolerance) {
                return false;
            }
        }
    }
    for (j = 0; j < solver->grid.extent[DIRECTION_Y]; ++j) {
        for (i = 0; i < solver->grid.extent[DIRECTION_X]; ++i) {
            const size_t lower = grid_index(&solver->grid, i, j, 0);
            const size_t upper = grid_index(&solver->grid, i, j, last_z);

            if (fabs(solver->state.velocity.component[DIRECTION_Z]
                         .data[lower]) > tolerance ||
                fabs(solver->state.velocity.component[DIRECTION_Z]
                         .data[upper]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

static bool flow_crosses_expanding_cavity(const Solver *solver)
{
    const Real tolerance = sizeof(Real) == sizeof(float)
        ? (Real)1e-7 : (Real)1e-14;
    const size_t center_y = solver->grid.extent[DIRECTION_Y] / 2;
    const size_t center_z = solver->grid.extent[DIRECTION_Z] / 2;
    const size_t narrow_section = grid_index(
        &solver->grid, solver->grid.extent[DIRECTION_X] / 4,
        center_y, center_z);
    const size_t expanded_section = grid_index(
        &solver->grid, 3 * solver->grid.extent[DIRECTION_X] / 4,
        center_y, center_z);

    return fabs(solver->state.velocity.component[DIRECTION_X]
                    .data[narrow_section]) > tolerance &&
           fabs(solver->state.velocity.component[DIRECTION_X]
                    .data[expanded_section]) > tolerance;
}

static bool flow_reaches_tunnel_interior(const Solver *solver)
{
    const Real tolerance = sizeof(Real) == sizeof(float)
        ? (Real)1e-7 : (Real)1e-14;
    const size_t index = grid_index(
        &solver->grid,
        solver->grid.extent[DIRECTION_X] / 4,
        solver->grid.extent[DIRECTION_Y] / 2,
        solver->grid.extent[DIRECTION_Z] / 2);

    return fabs(solver->state.velocity.component[DIRECTION_X].data[index]) >
           tolerance;
}

static bool run_flow_case(const ProblemDefinition *problem,
                          bool cavity_case,
                          bool output_enabled)
{
    SolverConfig config = solver_default_config();
    Solver solver = {0};
    SolverStatus status;
    bool success = false;

    config.extent[DIRECTION_X] = cavity_case ? 32 : 48;
    config.extent[DIRECTION_Y] = 32;
    config.extent[DIRECTION_Z] = 32;
    config.domain_length[DIRECTION_X] =
        cavity_case ? CAVITY_SIDE : TUNNEL_LENGTH_X;
    config.domain_length[DIRECTION_Y] =
        cavity_case ? CAVITY_SIDE : TUNNEL_LENGTH_Y;
    config.domain_length[DIRECTION_Z] =
        cavity_case ? CAVITY_SIDE : TUNNEL_LENGTH_Z;
    config.dt = (Real)0.001;
    config.steps = 2000;
    config.viscosity = (Real)0.05;
    config.output_frequency = output_enabled ? 5 : 0;
    config.output_directory = output_enabled ? "output" : NULL;

    status = solver_init(&solver, &config, problem);
    if (status != SOLVER_SUCCESS) {
        fprintf(stderr, "%s initialization failed\n", problem->name);
        goto cleanup;
    }
    if (!gamma_contains_fluid_and_solid(&solver)) {
        fprintf(stderr, "%s permeability is incorrect\n", problem->name);
        goto cleanup;
    }
    status = solver_solve(&solver);
    if (status != SOLVER_SUCCESS ||
        solver.stats.completed_steps != config.steps) {
        fprintf(stderr, "%s solve failed\n", problem->name);
        goto cleanup;
    }
    if (!state_is_finite(&solver)) {
        fprintf(stderr, "%s produced non-finite state\n", problem->name);
        goto cleanup;
    }
    if (!open_faces_have_prescribed_velocity(&solver)) {
        fprintf(stderr, "%s inlet or outlet is incorrect\n", problem->name);
        goto cleanup;
    }
    if (!wall_normal_velocity_is_zero(&solver)) {
        fprintf(stderr, "%s wall-normal velocity is not zero\n",
                problem->name);
        goto cleanup;
    }
    if (cavity_case && !flow_crosses_expanding_cavity(&solver)) {
        fprintf(stderr, "%s flow did not cross the cavity\n", problem->name);
        goto cleanup;
    }
    if (!cavity_case && !flow_reaches_tunnel_interior(&solver)) {
        fprintf(stderr, "%s flow did not reach the tunnel\n", problem->name);
        goto cleanup;
    }
    success = true;

cleanup:
    solver_destroy(&solver);
    return success;
}

int main(int argc, char **argv)
{
    static const ProblemDefinition tunnel_problem = {
        "tunnel with a variable-permeability obstacle",
        zero_velocity,
        zero_pressure,
        tunnel_boundary_velocity,
        zero_velocity,
        tunnel_permeability
    };
    static const ProblemDefinition cavity_problem = {
        "square inlet and expanding cavity tunnel",
        zero_velocity,
        zero_pressure,
        cavity_boundary_velocity,
        zero_velocity,
        cavity_permeability
    };
    bool run_tunnel = true;
    bool run_cavity = true;
    bool output_enabled = true;
    int argument;

    for (argument = 1; argument < argc; ++argument) {
        if (strcmp(argv[argument], "--no-output") == 0) {
            output_enabled = false;
        } else if (strcmp(argv[argument], "--case") == 0 &&
                   argument + 1 < argc) {
            const char *selected_case = argv[++argument];
            if (strcmp(selected_case, "tunnel") == 0) {
                run_tunnel = true;
                run_cavity = false;
            } else if (strcmp(selected_case, "cavity") == 0) {
                run_tunnel = false;
                run_cavity = true;
            } else if (strcmp(selected_case, "all") == 0) {
                run_tunnel = true;
                run_cavity = true;
            } else {
                fprintf(stderr, "case must be tunnel, cavity, or all\n");
                return 1;
            }
        } else {
            fprintf(stderr,
                    "usage: %s [--case tunnel|cavity|all] [--no-output]\n",
                    argv[0]);
            return 1;
        }
    }

    if (run_tunnel &&
        !run_flow_case(&tunnel_problem, false, output_enabled)) return 1;
    if (run_cavity &&
        !run_flow_case(&cavity_problem, true, output_enabled)) return 1;
    return 0;
}
