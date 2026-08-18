#include "solver.h"

#include <math.h>

#ifdef USE_FLOAT
#define REAL_SIN sinf
#define REAL_COS cosf
#else
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

Real porosity_fn(Real x, Real y, Real z, Real t, int component) {
    (void)x;
    (void)y;
    (void)z;
    (void)t;

    switch(component) {
        case 0: return 1.0;
        case 1: return 1.0;
        case 2: return 1.0;
        default: return 0.0;
    }
}

/* Exact velocity values at t = 0 */
Real paper_velocity_fn(Real x, Real y, Real z, Real t, int component) {
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

/* Exact pressure values at t = 1/2 */ // TODO:Should I set t = 1/2 ?
Real paper_pressure_fn(Real x, Real y, Real z, Real t) {
    return - 3.0f * NU * REAL_COS(x) * REAL_COS(t + y) * REAL_COS(z);
}

const Data paper_data = {
    .name = "paper_data",
    .bc_velocity = paper_bc_velocity,
    .forcing_fn  = paper_forcing,
    .porosity_fn = porosity_fn,
    .velocity_fn = paper_velocity_fn,
    .pressure_fn = paper_pressure_fn,
};


// TODO ? const Data *data_table; // list of Data existing in the solver, needed for parsing
// const Data zero_pressure_data = {};
