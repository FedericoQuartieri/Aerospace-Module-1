/* Three-dimensional channel with a wall-attached inclined Brinkman obstacle. */
#include "solver.h"

#define MAX_STREAMWISE_SPEED ((Real)1.0)
#define FREE_FLUID_PERMEABILITY ((Real)1e30)

#define SOLID_PERMEABILITY ((Real)1e-4) // Pay attention to instability
#define OBSTACLE_ANGLE ((Real)(35.0 * M_PI / 180.0))
#define OBSTACLE_HALF_LENGTH ((Real)0.40 * (Real)LY)
#define OBSTACLE_HALF_THICKNESS ((Real)0.04 * (Real)LY)

static Real zero_forcing(Real x, Real y, Real z, Real t, int component)
{
    (void)x;
    (void)y;
    (void)z;
    (void)t;
    (void)component;

    return (Real)0;
}

static Real zero_pressure(Real x, Real y, Real z, Real t)
{
    (void)x;
    (void)y;
    (void)z;
    (void)t;

    return (Real)0;
}

/*
 * Profile of inlet and outlet cross-sections:
 * zero along the four lateral walls and reaches MAX_STREAMWISE_SPEED at the
 * center of the section.
 */
static Real streamwise_profile(Real y, Real z)
{
    const Real normalized_y = y / (Real)LY;
    const Real normalized_z = z / (Real)LZ;

    if (normalized_y < (Real)0 || normalized_y > (Real)1 ||
        normalized_z < (Real)0 || normalized_z > (Real)1) {
        return (Real)0;
    }

    return (Real)16 * MAX_STREAMWISE_SPEED *
           normalized_y * ((Real)1 - normalized_y) *
           normalized_z * ((Real)1 - normalized_z);
}

/*
 * Fluid enters through x == 0 and leaves through x == LX.  Prescribing the
 * same profile on the two opposite sections gives equal inlet and outlet
 * flow rates.  All the lateral walls have zero velocity.
 */
static Real channel_boundary_velocity(Real x,
                                      Real y,
                                      Real z,
                                      Real t,
                                      int component)
{
    const Real face_tolerance = (Real)DX / (Real)4;
    const int on_streamwise_section =
        x < face_tolerance || x > (Real)LX - face_tolerance;

    (void)t;

    return component == 0 && on_streamwise_section
        ? streamwise_profile(y, z)
        : (Real)0;
}

/* Initial condition compatible with the inlet and outlet profiles. */
static Real channel_initial_velocity(Real x,
                                     Real y,
                                     Real z,
                                     Real t,
                                     int component)
{
    (void)x;
    (void)t;

    return component == 0
        ? streamwise_profile(y, z)
        : (Real)0;
}

/*
 * Rectangular obstacle rotated by OBSTACLE_ANGLE and extruded through the
 * complete Z direction.  Its lower centerline endpoint is attached to the
 * y == 0 wall, while its center remains halfway along the X direction.
 */
static Real inclined_obstacle_permeability(Real x,
                                           Real y,
                                           Real z,
                                           Real t,
                                           int component)
{
    const Real cosine = (Real)cos((double)OBSTACLE_ANGLE);
    const Real sine = (Real)sin((double)OBSTACLE_ANGLE);
    const Real obstacle_center_y = OBSTACLE_HALF_LENGTH * sine;
    const Real relative_x = x - (Real)LX / (Real)2;
    const Real relative_y = y - obstacle_center_y;
    const Real along_obstacle =
        cosine * relative_x + sine * relative_y;
    const Real across_obstacle =
        -sine * relative_x + cosine * relative_y;
    const int inside_obstacle =
        fabs((double)along_obstacle) <=
            (double)OBSTACLE_HALF_LENGTH &&
        fabs((double)across_obstacle) <=
            (double)OBSTACLE_HALF_THICKNESS;

    (void)z;
    (void)t;
    (void)component;

    return inside_obstacle
        ? SOLID_PERMEABILITY
        : FREE_FLUID_PERMEABILITY;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    Data data = {
        .name = "Channel with wall-attached inclined Brinkman obstacle",
        .bc_velocity = channel_boundary_velocity,
        .forcing_fn = zero_forcing,
        .porosity_fn = inclined_obstacle_permeability,
        .velocity_fn = channel_initial_velocity,
        .pressure_fn = zero_pressure,
    };
    SolverMemState state;
    SolverStats stats = {0};
    const int write_enabled = 1;

    solver_init(&state, &data, NULL);
    solver_solve(&state, &data, &stats, write_enabled);

    solver_destroy(&state);
    MPI_Finalize();

    return 0;
}
