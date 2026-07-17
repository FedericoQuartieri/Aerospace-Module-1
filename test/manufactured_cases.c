#include "manufactured_cases.h"

#include <math.h>

#define PI_VALUE ((Real)M_PI)
#define BASE_CONFIG_INITIALIZER \
    {{64, 64, 64}, {PI_VALUE, PI_VALUE, PI_VALUE}, \
     (Real)0.0007, 10, (Real)1, 0, NULL}

static Real paper_velocity(Real x,
                           Real y,
                           Real z,
                           Real time,
                           Direction component)
{
    switch (component) {
        case DIRECTION_X: return sin(x) * cos(time + y) * sin(z);
        case DIRECTION_Y: return cos(x) * sin(time + y) * sin(z);
        case DIRECTION_Z:
            return (Real)2 * cos(x) * cos(time + y) * cos(z);
        default: return (Real)0;
    }
}

static Real paper_pressure(Real x, Real y, Real z, Real time)
{
    return (Real)-3 * cos(x) * cos(time + y) * cos(z);
}

static Real zero_pressure(Real x, Real y, Real z, Real time)
{
    (void)x;
    (void)y;
    (void)z;
    (void)time;
    return (Real)0;
}

static Real unit_permeability(Real x, Real y, Real z, Real time)
{
    (void)x;
    (void)y;
    (void)z;
    (void)time;
    return (Real)1;
}

static Real variable_permeability(Real x, Real y, Real z, Real time)
{
    (void)time;
    return (Real)1 + (Real)0.25 * sin(x) * sin(y) * sin(z);
}

static Real forcing_with_pressure_and_permeability(
    Real x,
    Real y,
    Real z,
    Real time,
    Direction component,
    ScalarFunction permeability,
    bool include_pressure)
{
    const Real velocity = paper_velocity(x, y, z, time, component);
    const Real k = permeability(x, y, z, time);
    Real time_derivative;
    Real pressure_gradient = (Real)0;

    switch (component) {
        case DIRECTION_X:
            time_derivative = -sin(x) * sin(time + y) * sin(z);
            if (include_pressure) {
                pressure_gradient =
                    (Real)3 * sin(x) * cos(time + y) * cos(z);
            }
            break;
        case DIRECTION_Y:
            time_derivative = cos(x) * cos(time + y) * sin(z);
            if (include_pressure) {
                pressure_gradient =
                    (Real)3 * cos(x) * sin(time + y) * cos(z);
            }
            break;
        case DIRECTION_Z:
            time_derivative =
                (Real)-2 * cos(x) * sin(time + y) * cos(z);
            if (include_pressure) {
                pressure_gradient =
                    (Real)3 * cos(x) * cos(time + y) * sin(z);
            }
            break;
        default:
            return (Real)0;
    }

    /* Each velocity component is an eigenfunction with Laplacian -3 u. */
    return time_derivative + (Real)3 * velocity + velocity / k +
           pressure_gradient;
}

static Real paper_forcing(Real x,
                          Real y,
                          Real z,
                          Real time,
                          Direction component)
{
    return forcing_with_pressure_and_permeability(
        x, y, z, time, component, unit_permeability, true);
}

static Real zero_pressure_forcing(Real x,
                                  Real y,
                                  Real z,
                                  Real time,
                                  Direction component)
{
    return forcing_with_pressure_and_permeability(
        x, y, z, time, component, unit_permeability, false);
}

static Real variable_permeability_forcing(Real x,
                                          Real y,
                                          Real z,
                                          Real time,
                                          Direction component)
{
    return forcing_with_pressure_and_permeability(
        x, y, z, time, component, variable_permeability, true);
}

const ManufacturedCase MANUFACTURED_CASES[] = {
    {
        {"paper", paper_velocity, paper_pressure, paper_velocity,
         paper_forcing, unit_permeability},
        BASE_CONFIG_INITIALIZER,
        paper_velocity,
        paper_pressure,
        (Real)3.2e-4,
        (Real)6.0e-4,
        (Real)1.1e-2,
        (Real)4.0e-2,
        (Real)2.0e-3,
        (Real)6.0e-4,
        (Real)1.70,
        (Real)1.95,
        (Real)1.80,
        (Real)1.35
    },
    {
        {"zero-pressure", paper_velocity, zero_pressure, paper_velocity,
         zero_pressure_forcing, unit_permeability},
        BASE_CONFIG_INITIALIZER,
        paper_velocity,
        zero_pressure,
        (Real)3.2e-4,
        (Real)6.0e-4,
        (Real)1.1e-2,
        (Real)4.0e-2,
        (Real)2.0e-3,
        (Real)6.0e-4,
        (Real)1.70,
        (Real)1.95,
        (Real)1.80,
        (Real)1.35
    },
    {
        {"variable-permeability", paper_velocity, paper_pressure,
         paper_velocity, variable_permeability_forcing,
         variable_permeability},
        BASE_CONFIG_INITIALIZER,
        paper_velocity,
        paper_pressure,
        (Real)3.0e-3,
        (Real)5.0e-3,
        (Real)8.0e-2,
        (Real)1.5e-1,
        (Real)2.0e-2,
        (Real)5.0e-3,
        (Real)0,
        (Real)0,
        (Real)0,
        (Real)0
    }
};

const size_t MANUFACTURED_CASE_COUNT =
    sizeof(MANUFACTURED_CASES) / sizeof(MANUFACTURED_CASES[0]);
