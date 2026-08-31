#include "solver.h"
#include "error_norms.h"

#ifdef USE_FLOAT
#define REAL_EXP expf
#define REAL_SIN sinf
#else
#define REAL_EXP exp
#define REAL_SIN sin
#endif

/*
 * Manufactured solution of the unsteady Stokes--Brinkman equation:
 *
 *   du/dt - NU Laplacian(u) + (NU/K)u + grad(p) = F,  div(u) = 0.
 *
 * The field v = (sin(y), sin(z), sin(x)) is divergence-free and satisfies
 * Laplacian(v) = -v.  Its exponential decay balances diffusion and
 * Brinkman drag, while the linear pressure produces the constant force F.
 */

#define K0 ((Real)1.0)

#define FORCE_X ((Real)1.0)
#define FORCE_Y ((Real)2.0)
#define FORCE_Z ((Real)3.0)

static Real manufactured_velocity(Real x,
                                  Real y,
                                  Real z,
                                  Real t,
                                  int component)
{
    const Real decay_rate =
        (Real)NU * ((Real)1 + (Real)1 / K0);
    const Real amplitude = REAL_EXP(-decay_rate * t);

    switch (component) {
    case 0:
        return amplitude * REAL_SIN(y);
    case 1:
        return amplitude * REAL_SIN(z);
    case 2:
        return amplitude * REAL_SIN(x);
    default:
        return (Real)0;
    }
}

static Real manufactured_pressure(Real x, Real y, Real z, Real t)
{
    (void)t;

    return FORCE_X * x + FORCE_Y * y + FORCE_Z * z;
}

static Real manufactured_constant_forcing(Real x,
                                          Real y,
                                          Real z,
                                          Real t,
                                          int component)
{
    (void)x;
    (void)y;
    (void)z;
    (void)t;

    switch (component) {
    case 0:
        return FORCE_X;
    case 1:
        return FORCE_Y;
    case 2:
        return FORCE_Z;
    default:
        return (Real)0;
    }
}

static Real manufactured_constant_permeability(Real x,
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

    return K0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    Data data = {
        .name = "Constant forcing sine",
        .bc_velocity = manufactured_velocity,
        .forcing_fn = manufactured_constant_forcing,
        .porosity_fn = manufactured_constant_permeability,
        .velocity_fn = manufactured_velocity,
        .pressure_fn = manufactured_pressure,
    };
    SolverMemState solver_mem_state;
    SolverStats solver_stats = {0};
    const int write_enabled = 0;

    solver_init(&solver_mem_state, &data, NULL);
    solver_solve(&solver_mem_state, &data, &solver_stats, write_enabled);

    const Real velocity_verification_time = (Real)STEPS * (Real)DT;
    const Real pressure_verification_time = velocity_verification_time - (Real)DT / (Real)2;
    const SolverErrorNorms errors = compute_solver_error_norms(&solver_mem_state,
                                   &data,
                                   velocity_verification_time,
                                   pressure_verification_time);

    print_solver_error_norms(&errors, velocity_verification_time, pressure_verification_time);

    solver_destroy(&solver_mem_state);
    MPI_Finalize();

    return 0;
}
