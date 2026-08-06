#include "solver.h"
#include "error_norms.h"

#ifdef USE_FLOAT
#define REAL_SIN sinf
#define REAL_COS cosf
#else
#define REAL_SIN sin
#define REAL_COS cos
#endif


static Real manufactured_zero_velocity(Real x,
                                       Real y,
                                       Real z,
                                       Real t,
                                       int component)
{
    switch (component) {
    case 0:
        return REAL_SIN(x) * REAL_COS(t + y) * REAL_SIN(z);
    case 1:
        return REAL_COS(x) * REAL_SIN(t + y) * REAL_SIN(z);
    case 2:
        return 2.0f * REAL_COS(x) * REAL_COS(t + y) * REAL_COS(z);
    default:
        return 0.0;
    }
}

static Real manufactured_zero_pressure(Real x, Real y, Real z, Real t)
{
    (void)x;
    (void)y;
    (void)z;
    (void)t;

    return 0.0;
}

static Real manufactured_zero_forcing(Real x,
                                      Real y,
                                      Real z,
                                      Real t,
                                      int component)
{
    const Real u_x = manufactured_zero_velocity(x, y, z, t, 0);
    const Real u_y = manufactured_zero_velocity(x, y, z, t, 1);
    const Real u_z = manufactured_zero_velocity(x, y, z, t, 2);

    const Real dudt_x =
        -REAL_SIN(x) * REAL_SIN(t + y) * REAL_SIN(z);
    const Real dudt_y =
        REAL_COS(x) * REAL_COS(t + y) * REAL_SIN(z);
    const Real dudt_z =
        -2.0 * REAL_COS(x) * REAL_SIN(t + y) * REAL_COS(z);

    const Real lap_u_x = -3.0 * u_x;
    const Real lap_u_y = -3.0 * u_y;
    const Real lap_u_z = -3.0 * u_z;

    const Real k = 1.0;
    const Real dpdx = 0.0;
    const Real dpdy = 0.0;
    const Real dpdz = 0.0;

    switch (component) {
    case 0:
        return dudt_x - NU * lap_u_x + (NU / k) * u_x + dpdx;
    case 1:
        return dudt_y - NU * lap_u_y + (NU / k) * u_y + dpdy;
    case 2:
        return dudt_z - NU * lap_u_z + (NU / k) * u_z + dpdz;
    default:
        return 0.0;
    }
}

static Real manufactured_unit_porosity(Real x,
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

    return 1.0;
}

int main(void)
{
    /* Manufactured solution corresponding to identically zero pressure. */
    Data data = {
        .name = "Zero pressure paper",
        .bc_velocity = manufactured_zero_velocity,
        .forcing_fn = manufactured_zero_forcing,
        .porosity_fn = manufactured_unit_porosity,
        .porosity_time_dependent = 0,
        .velocity_fn = manufactured_zero_velocity,
        .pressure_fn = manufactured_zero_pressure,
    };
    SolverMemState solver_mem_state;
    SolverStats solver_stats = {0};
    Decomp decomp;

    decomp_init_serial(&decomp);
    solver_init(&decomp, &solver_mem_state, &data, NULL);
    
    int write_enabled = 0;
    solver_solve(&decomp, &solver_mem_state, &data, &solver_stats,
                 write_enabled);

    const Real velocity_verification_time =
        (Real)STEPS * (Real)DT;
    const Real pressure_verification_time =
        velocity_verification_time - (Real)DT / 2.0;
    const SolverErrorNorms errors =
        compute_solver_error_norms(&decomp,
                                   &solver_mem_state,
                                   &data,
                                   velocity_verification_time,
                                   pressure_verification_time);

    print_solver_error_norms(&decomp,
                             &errors,
                             velocity_verification_time,
                             pressure_verification_time);

    return 0;
}
