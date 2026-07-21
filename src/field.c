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

void vectorField_fill(VectorField *vf, VectorFunction vector_fn) {

}

void scalarField_fill(ScalarField *sf, ScalarFunction scalar_fn) {

}
