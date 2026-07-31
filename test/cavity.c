#include "solver.h"

#define LID_SPEED ((Real)1.0)
#define FREE_FLUID_PERMEABILITY ((Real)1e30)

static Real zero_vector(Real x, Real y, Real z, Real t, int component)
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
 * Three-dimensional lid-driven cavity:
 * - the upper Y wall moves in the positive X direction;
 * - all the other walls are stationary.
 *
 * The tolerance separates the physical wall at y == LY from the closest
 * staggered interior point, located half a cell below it.
 */
static Real cavity_boundary_velocity(Real x,
                                     Real y,
                                     Real z,
                                     Real t,
                                     int component)
{
    const int on_lid = y > (Real)LY - (Real)DY / (Real)4;

    (void)x;
    (void)z;
    (void)t;

    return component == 0 && on_lid ? LID_SPEED : (Real)0;
}

/*
 * The solver uses this field as the Brinkman permeability K in the term
 * NU/K.  A large, finite value represents a cavity filled by free fluid.
 */
static Real free_fluid_permeability(Real x,
                                    Real y,
                                    Real z,
                                    Real t,
                                    int component)
{
    (void)x;
    (void)y;
    (void)z;
    (void)t;
    (void)component;

    return FREE_FLUID_PERMEABILITY;
}

int main(void)
{
    Data data = {
        .name = "Lid-driven cavity",
        .bc_velocity = cavity_boundary_velocity,
        .forcing_fn = zero_vector,
        .porosity_fn = free_fluid_permeability,
        .velocity_fn = zero_vector,
        .pressure_fn = zero_pressure,
    };
    SolverMemState state;
    SolverStats stats = {0};
    const int write_enabled = 1;

    solver_init(&state, &data, NULL);
    solver_solve(&state, &data, &stats, write_enabled);

    return 0;
}
