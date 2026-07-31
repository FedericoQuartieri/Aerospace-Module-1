#include "field.h"
#include "solver.h"
#include "utils.h"

/* Allocate the array of cells for a scalar field */
void scalarField_alloc(ScalarField *sf) {
   size_t field_size = GRID_CELLS * sizeof(Real);
   sf->v = xmalloc(field_size);
}

/* Allocate the array of cells for each component of a vector field */
void vectorField_alloc(VectorField *vf) {
    size_t field_size = GRID_CELLS * sizeof(Real);
    vf->v_x = xmalloc(field_size);
    vf->v_y = xmalloc(field_size);
    vf->v_z = xmalloc(field_size);
}

void vectorField_fill(VectorField *restrict vf,
                      VectorFunction vector_fn,
                      Real t_step) {
    Real time = time_physical_coord(t_step);
    Real *restrict v_x = vf->v_x;
    Real *restrict v_y = vf->v_y;
    Real *restrict v_z = vf->v_z;
    size_t off = 0;

    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                Real x = centered_physical_coord(i, 0);
                Real y = centered_physical_coord(j, 1);
                Real z = centered_physical_coord(k, 2);

                v_x[off] =
                    vector_fn(staggered_physical_coord(i, 0),
                              y, z, time, 0);
                v_y[off] =
                    vector_fn(x, staggered_physical_coord(j, 1),
                              z, time, 1);
                v_z[off] =
                    vector_fn(x, y,
                              staggered_physical_coord(k, 2),
                              time, 2);
                off++;
            }
        }
    }
}

void scalarField_fill(ScalarField *restrict sf,
                      ScalarFunction scalar_fn,
                      Real t_step) {
    Real time = time_physical_coord(t_step);
    Real *restrict v = sf->v;
    size_t off = 0;

    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                Real x = centered_physical_coord(i, 0);
                Real y = centered_physical_coord(j, 1);
                Real z = centered_physical_coord(k, 2);

                v[off] = scalar_fn(x, y, z, time);
                off++;
            }
        }
    }
}
