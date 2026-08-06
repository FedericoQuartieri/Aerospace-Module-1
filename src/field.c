#include "field.h"
#include "solver.h"
#include "utils.h"

/* Allocate the array of cells for a scalar field */
void scalarField_alloc(const Decomp *d, ScalarField *sf) {
   size_t field_size = d->n_cells * sizeof(Real);
   sf->v = xmalloc(field_size);
}

/* Allocate the array of cells for each component of a vector field */
void vectorField_alloc(const Decomp *d, VectorField *vf) {
    size_t field_size = d->n_cells * sizeof(Real);
    vf->v_x = xmalloc(field_size);
    vf->v_y = xmalloc(field_size);
    vf->v_z = xmalloc(field_size);
}

/*
 * The sampled functions describe the physical problem, so they are evaluated
 * at the *global* position of each cell.  Only the loop bounds and the memory
 * offset use local indices.
 */
void vectorField_fill(const Decomp *restrict d,
                      VectorField *restrict vf,
                      VectorFunction vector_fn,
                      Real t_step) {
    Real time = time_physical_coord(t_step);
    Real *restrict v_x = vf->v_x;
    Real *restrict v_y = vf->v_y;
    Real *restrict v_z = vf->v_z;

    for (int k = 0; k < d->n[2]; k++) {
        int gk = decomp_global(d, k, 2);

        for (int j = 0; j < d->n[1]; j++) {
            int gj = decomp_global(d, j, 1);
            size_t row = decomp_index(d, 0, j, k);

            for (int i = 0; i < d->n[0]; i++) {
                int gi = decomp_global(d, i, 0);
                Real x = centered_physical_coord(gi, 0);
                Real y = centered_physical_coord(gj, 1);
                Real z = centered_physical_coord(gk, 2);
                size_t off = row + (size_t)i;

                v_x[off] =
                    vector_fn(staggered_physical_coord(gi, 0),
                              y, z, time, 0);
                v_y[off] =
                    vector_fn(x, staggered_physical_coord(gj, 1),
                              z, time, 1);
                v_z[off] =
                    vector_fn(x, y,
                              staggered_physical_coord(gk, 2),
                              time, 2);
            }
        }
    }
}

void scalarField_fill(const Decomp *restrict d,
                      ScalarField *restrict sf,
                      ScalarFunction scalar_fn,
                      Real t_step) {
    Real time = time_physical_coord(t_step);
    Real *restrict v = sf->v;

    for (int k = 0; k < d->n[2]; k++) {
        int gk = decomp_global(d, k, 2);

        for (int j = 0; j < d->n[1]; j++) {
            int gj = decomp_global(d, j, 1);
            size_t row = decomp_index(d, 0, j, k);

            for (int i = 0; i < d->n[0]; i++) {
                int gi = decomp_global(d, i, 0);
                Real x = centered_physical_coord(gi, 0);
                Real y = centered_physical_coord(gj, 1);
                Real z = centered_physical_coord(gk, 2);

                v[row + (size_t)i] = scalar_fn(x, y, z, time);
            }
        }
    }
}
