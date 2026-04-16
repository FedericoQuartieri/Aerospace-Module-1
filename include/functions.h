/* manufactured_solution.h */
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "constants.h"

typedef DTYPE (*vector_component_fn)(DTYPE x, DTYPE y, DTYPE z, DTYPE t, int component);


typedef struct {
    const char *name;

    /* velocity boundary conditions */
    vector_component_fn bc_velocity;

    /* forcing component */
    vector_component_fn forcing;
} Data;


static inline DTYPE bc_velocity(
    const Data *data,
    DTYPE x, DTYPE y, DTYPE z, DTYPE t,
    int component)
{
    return data->bc_velocity(x, y, z, t, component);
}

static inline DTYPE forcing(
    const Data *data,
    DTYPE x, DTYPE y, DTYPE z, DTYPE t,
    int component)
{
    return data->forcing(x, y, z, t, component);
}

static inline DTYPE delta_bc_velocity(
    const Data *data,
    DTYPE x, DTYPE y, DTYPE z, DTYPE t,
    int component)
{
    if (t == 0.0) {
        return bc_velocity(data, x, y, z, t, component);
    }

    return bc_velocity(data, x, y, z, t, component)
         - bc_velocity(data, x, y, z, t - DT, component);
}

#endif