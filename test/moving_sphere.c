/* Three-dimensional channel with a moving spherical Brinkman obstacle. */
#include "solver.h"
#include "parallel.h"

#define MAX_STREAMWISE_SPEED ((Real)1.0)
#define FREE_FLUID_PERMEABILITY ((Real)1e30)
#define SOLID_PERMEABILITY ((Real)2e-3)
#define SPHERE_RADIUS ((Real)0.18 * (Real)LY)
#define SPHERE_TRAVEL_AMPLITUDE ((Real)0.25 * (Real)LX)

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
 * Fluid enters through x == 0 and leaves through x == LX with the same
 * profile.  The four lateral walls are stationary.
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
 * The sphere starts at the channel center and completes one sinusoidal
 * round trip along X during the simulated interval [0, T].
 */
static Real sphere_center_x(Real t)
{
    const Real phase = (Real)2 * (Real)M_PI * t / (Real)T;

    return (Real)LX / (Real)2 +
           SPHERE_TRAVEL_AMPLITUDE * (Real)sin((double)phase);
}

static Real moving_sphere_permeability(Real x,
                                       Real y,
                                       Real z,
                                       Real t,
                                       int component)
{
    const Real relative_x = x - sphere_center_x(t);
    const Real relative_y = y - (Real)LY / (Real)2;
    const Real relative_z = z - (Real)LZ / (Real)2;
    const Real squared_distance =
        relative_x * relative_x +
        relative_y * relative_y +
        relative_z * relative_z;

    (void)component;

    return squared_distance <= SPHERE_RADIUS * SPHERE_RADIUS
        ? SOLID_PERMEABILITY
        : FREE_FLUID_PERMEABILITY;
}

int main(void)
{
    par_init(NULL, NULL);

    Data data = {
        .name = "Channel with moving spherical obstacle",
        .bc_velocity = channel_boundary_velocity,
        .forcing_fn = zero_forcing,
        .porosity_fn = moving_sphere_permeability,
        .porosity_time_dependent = 1,
        .velocity_fn = channel_initial_velocity,
        .pressure_fn = zero_pressure,
    };
    SolverMemState state;
    SolverStats stats = {0};
    const int write_enabled = 1;
    Decomp decomp;

    /* {1, 1, 0}: i processi si dispongono a fette lungo Z. */
    const int process_grid[3] = {1, 1, 0};
    par_topology_init(process_grid);
    decomp_init_mpi(&decomp);
    solver_init(&decomp, &state, &data, NULL);
    solver_solve(&decomp, &state, &data, &stats, write_enabled);

    par_finalize();
    return 0;
}
