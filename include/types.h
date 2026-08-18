#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

#ifdef USE_FLOAT
typedef float Real;
#else
typedef double Real;
#endif

/* Function of space, time and vector component (0: X, 1: Y, 2: Z). */
typedef Real (*VectorFunction)(Real x, Real y, Real z, Real t, int component);

/* Function of space and time */
typedef Real (*ScalarFunction)(Real x, Real y, Real z, Real t);

typedef struct Data {
    const char *name;
    VectorFunction bc_velocity;
    VectorFunction forcing_fn;
    VectorFunction porosity_fn;
    VectorFunction velocity_fn;
    ScalarFunction pressure_fn;
} Data;

typedef struct ScalarField {
    Real *v;
} ScalarField;

typedef struct VectorField {
    Real *v_x;
    Real *v_y;
    Real *v_z;
} VectorField;

typedef struct SolverStats {
    /* Accumulated execution times, in nanoseconds. */
    uint64_t momentum_halo;
    uint64_t eta_sys;
    uint64_t zeta_sys;
    uint64_t u_sys;
    uint64_t psi_sys;
    uint64_t phi_low_sys;
    uint64_t phi_high_sys;
    uint64_t pressure_halo;
    uint64_t pressure_update;
    uint64_t solve_steps;
    uint64_t wr_output;
} SolverStats;

#endif
