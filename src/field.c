#include "field.h"
#include "solver.h"
#include "utils.h"

/* Allocate the array of cells for a scalar field */
void scalarField_alloc(ScalarField *sf, const Domain *domain) {
   size_t field_size = domain->allocated_cells * sizeof(Real);
   sf->v = xmalloc(field_size);
}

/* Allocate the array of cells for each component of a vector field */
void vectorField_alloc(VectorField *vf, const Domain *domain) {
    size_t field_size = domain->allocated_cells * sizeof(Real);
    vf->v_x = xmalloc(field_size);
    vf->v_y = xmalloc(field_size);
    vf->v_z = xmalloc(field_size);
}

void vectorField_fill(VectorField *restrict vf,
                      const Domain *domain,
                      VectorFunction vector_fn,
                      Real t_step) {
    Real time = time_physical_coord(t_step);
    Real *restrict v_x = vf->v_x;
    Real *restrict v_y = vf->v_y;
    Real *restrict v_z = vf->v_z;
    for (int k = 0; k < domain->local[AXIS_Z]; k++) {
        int global_k = domain_global_index(domain, k, AXIS_Z);
        for (int j = 0; j < domain->local[AXIS_Y]; j++) {
            int global_j = domain_global_index(domain, j, AXIS_Y);
            for (int i = 0; i < domain->local[AXIS_X]; i++) {
                int global_i = domain_global_index(domain, i, AXIS_X);
                size_t off = domain_index(domain, i, j, k);
                Real x = centered_physical_coord(global_i, 0);
                Real y = centered_physical_coord(global_j, 1);
                Real z = centered_physical_coord(global_k, 2);

                v_x[off] =
                    vector_fn(staggered_physical_coord(global_i, 0),
                              y, z, time, 0);
                v_y[off] =
                    vector_fn(x, staggered_physical_coord(global_j, 1),
                              z, time, 1);
                v_z[off] =
                    vector_fn(x, y,
                              staggered_physical_coord(global_k, 2),
                              time, 2);
            }
        }
    }
}

void scalarField_fill(ScalarField *restrict sf,
                      const Domain *domain,
                      ScalarFunction scalar_fn,
                      Real t_step) {
    Real time = time_physical_coord(t_step);
    Real *restrict v = sf->v;
    for (int k = 0; k < domain->local[AXIS_Z]; k++) {
        int global_k = domain_global_index(domain, k, AXIS_Z);
        for (int j = 0; j < domain->local[AXIS_Y]; j++) {
            int global_j = domain_global_index(domain, j, AXIS_Y);
            for (int i = 0; i < domain->local[AXIS_X]; i++) {
                int global_i = domain_global_index(domain, i, AXIS_X);
                size_t off = domain_index(domain, i, j, k);
                Real x = centered_physical_coord(global_i, 0);
                Real y = centered_physical_coord(global_j, 1);
                Real z = centered_physical_coord(global_k, 2);

                v[off] = scalar_fn(x, y, z, time);
            }
        }
    }
}

void vectorField_free(VectorField *vf) {
    free(vf->v_x);
    free(vf->v_y);
    free(vf->v_z);
}

