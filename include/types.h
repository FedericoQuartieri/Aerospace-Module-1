#ifndef TYPES_H
#define TYPES_H

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

#endif
