/* Three-dimensional channel with an inclined Brinkman obstacle. */
#include "solver.h"

#define MAX_STREAMWISE_SPEED ((Real)1.0)
#define FREE_FLUID_PERMEABILITY ((Real)1e30)
/*
 * With DT = 1/300 and NU = 1, values below DT*NU/2 produce an oscillatory
 * Crank-Nicolson amplification factor for the Brinkman drag.  This value
 * gives a positive factor of about 0.09 while retaining a strong resistance.
 */
#define SOLID_PERMEABILITY ((Real)2e-3)
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
 * Fully developed profile on the inlet and outlet cross-sections.  It is
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
 * Rectangular obstacle centered in the X-Y plane, rotated by
 * OBSTACLE_ANGLE and extruded through the complete Z direction.
 *
 * The solver uses this field as the Brinkman permeability K in the term
 * NU/K.  The solid therefore has a small K (large resistance), while the
 * surrounding free fluid has a large K.
 */
static Real inclined_obstacle_permeability(Real x,
                                           Real y,
                                           Real z,
                                           Real t,
                                           int component)
{
    const Real relative_x = x - (Real)LX / (Real)2;
    const Real relative_y = y - (Real)LY / (Real)2;
    const Real cosine = (Real)cos((double)OBSTACLE_ANGLE);
    const Real sine = (Real)sin((double)OBSTACLE_ANGLE);
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

int main(void)
{
    Data data = {
        .name = "Channel with inclined Brinkman obstacle",
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

    return 0;
}
