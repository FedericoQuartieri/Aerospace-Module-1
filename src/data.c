#include "solver.h"

#include <math.h>
#include <string.h>

#ifdef USE_FLOAT
#define REAL_EXP expf
#define REAL_SIN sinf
#define REAL_COS cosf
#else
#define REAL_EXP exp
#define REAL_SIN sin
#define REAL_COS cos
#endif


/* In this example, the bc_velocity concide with the paper function, but in general 
 * it could be not known, tha's why we have two functions: bc_velocity and velocity_fn (t=0)
 * */
static Real paper_bc_velocity(Real x, Real y, Real z, Real t, int component)
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

static Real paper_forcing(Real x, Real y, Real z, Real t, int component)
{
    Real sin_ty;
    Real cos_ty;
    Real sin_z;
    Real cos_z;

    if (component < 0 || component > 2) {
        return 0.0;
    }

    sin_ty = REAL_SIN(t + y);
    cos_ty = REAL_COS(t + y);
    sin_z = REAL_SIN(z);
    cos_z = REAL_COS(z);

    switch (component) {
    case 0:
        return REAL_SIN(x)
               * (sin_z * (4.0f * cos_ty - sin_ty)
                  + 3.0f * cos_ty * cos_z);
    case 1:
        return REAL_COS(x)
               * (sin_z * (cos_ty + 4.0f * sin_ty)
                  + 3.0f * sin_ty * cos_z);
    case 2:
        return REAL_COS(x)
               * (cos_z * (8.0f * cos_ty - 2.0f * sin_ty)
                  + 3.0f * cos_ty * sin_z);
    default:
        return 0.0;
    }
}

static Real unit_porosity(Real x, Real y, Real z, Real t, int component) {
    (void)x;
    (void)y;
    (void)z;
    (void)t;
    (void)component;

    return (Real)1;
}

/* Exact velocity values at t = 0 */
static Real paper_velocity_fn(Real x, Real y, Real z, Real t, int component) {
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

/* Exact pressure value at the physical time passed by the caller. */
static Real paper_pressure_fn(Real x, Real y, Real z, Real t) {
    return - 3.0f * NU * REAL_COS(x) * REAL_COS(t + y) * REAL_COS(z);
}

static Real zero_pressure_fn(Real x, Real y, Real z, Real t)
{
    (void)x;
    (void)y;
    (void)z;
    (void)t;

    return (Real)0;
}

/*
 * Caso manufatto con pressione nulla: isola l'avanzamento della velocita' dal
 * gradiente di pressione, ed e' per questo un buon confronto rapido fra Schur
 * e pipeline.
 */
static Real zero_pressure_forcing(Real x, Real y, Real z, Real t,
                                  int component)
{
    const Real u_x = paper_velocity_fn(x, y, z, t, 0);
    const Real u_y = paper_velocity_fn(x, y, z, t, 1);
    const Real u_z = paper_velocity_fn(x, y, z, t, 2);

    const Real dudt_x =
        -REAL_SIN(x) * REAL_SIN(t + y) * REAL_SIN(z);
    const Real dudt_y =
        REAL_COS(x) * REAL_COS(t + y) * REAL_SIN(z);
    const Real dudt_z =
        -2.0 * REAL_COS(x) * REAL_SIN(t + y) * REAL_COS(z);

    const Real lap_u_x = -3.0 * u_x;
    const Real lap_u_y = -3.0 * u_y;
    const Real lap_u_z = -3.0 * u_z;

    switch (component) {
    case 0:
        return dudt_x - NU * lap_u_x + NU * u_x;
    case 1:
        return dudt_y - NU * lap_u_y + NU * u_y;
    case 2:
        return dudt_z - NU * lap_u_z + NU * u_z;
    default:
        return (Real)0;
    }
}

#define CONSTANT_K0 ((Real)1)
#define CONSTANT_FORCE_X ((Real)1)
#define CONSTANT_FORCE_Y ((Real)2)
#define CONSTANT_FORCE_Z ((Real)3)

static Real constant_forcing_velocity(Real x,
                                      Real y,
                                      Real z,
                                      Real t,
                                      int component)
{
    const Real decay_rate =
        (Real)NU * ((Real)1 + (Real)1 / CONSTANT_K0);
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

static Real constant_forcing_pressure(Real x, Real y, Real z, Real t)
{
    (void)t;

    return CONSTANT_FORCE_X * x +
           CONSTANT_FORCE_Y * y +
           CONSTANT_FORCE_Z * z;
}

static Real constant_forcing_fn(Real x,
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
        return CONSTANT_FORCE_X;
    case 1:
        return CONSTANT_FORCE_Y;
    case 2:
        return CONSTANT_FORCE_Z;
    default:
        return (Real)0;
    }
}

static Real constant_permeability(Real x,
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

    return CONSTANT_K0;
}

const Data paper_data = {
    .name = "paper_data",
    .bc_velocity = paper_bc_velocity,
    .forcing_fn  = paper_forcing,
    .porosity_fn = unit_porosity,
    .porosity_time_dependent = 0,
    .velocity_fn = paper_velocity_fn,
    .pressure_fn = paper_pressure_fn,
};

static const Data zero_pressure_data = {
    .name = "zero_pressure",
    .bc_velocity = paper_velocity_fn,
    .forcing_fn = zero_pressure_forcing,
    .porosity_fn = unit_porosity,
    .porosity_time_dependent = 0,
    .velocity_fn = paper_velocity_fn,
    .pressure_fn = zero_pressure_fn,
};

static const Data constant_forcing_data = {
    .name = "constant_forcing",
    .bc_velocity = constant_forcing_velocity,
    .forcing_fn = constant_forcing_fn,
    .porosity_fn = constant_permeability,
    .porosity_time_dependent = 0,
    .velocity_fn = constant_forcing_velocity,
    .pressure_fn = constant_forcing_pressure,
};


/*
 * Gli scenari che il solutore sa eseguire, cercati per nome.  Gli altri casi
 * (cavity, canale, sfera) vivono nei test, che costruiscono il proprio Data
 * direttamente e non passano da qui.
 */
static const Data *const data_table[] = {
    &paper_data,
    &zero_pressure_data,
    &constant_forcing_data,
};

const Data *data_by_name(const char *name) {
    size_t count = sizeof data_table / sizeof data_table[0];

    for (size_t i = 0; i < count; i++) {
        if (strcmp(data_table[i]->name, name) == 0) {
            return data_table[i];
        }
    }

    return NULL;
}

void data_print_names(FILE *stream) {
    size_t count = sizeof data_table / sizeof data_table[0];

    for (size_t i = 0; i < count; i++) {
        fprintf(stream, "  %s\n", data_table[i]->name);
    }
}
