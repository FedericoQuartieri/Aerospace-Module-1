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

void vectorField_fill(VectorField *vf, VectorFunction vector_fn, int t_step) {
    const Real t = time_physical_coord(t_step);
    size_t off = 0;

    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                const Real x = centered_physical_coord(i, 0);
                const Real y = centered_physical_coord(j, 1);
                const Real z = centered_physical_coord(k, 2);

                vf->v_x[off] =
                    vector_fn(staggered_physical_coord(i, 0),
                              y, z, t, 0);
                vf->v_y[off] =
                    vector_fn(x, staggered_physical_coord(j, 1),
                              z, t, 1);
                vf->v_z[off] =
                    vector_fn(x, y,
                              staggered_physical_coord(k, 2),
                              t, 2);
                off++;
            }
        }
    }
}

void scalarField_fill(ScalarField *sf, ScalarFunction scalar_fn, int t_step) {
    const Real t = time_physical_coord(t_step);
    size_t off = 0;
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                const Real x = centered_physical_coord(i, 0);
                const Real y = centered_physical_coord(j, 1);
                const Real z = centered_physical_coord(k, 2);

                sf->v[off] = scalar_fn(x, y, z, t);
                off++;
            }
        }
    }
}
