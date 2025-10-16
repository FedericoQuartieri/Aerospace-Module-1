#include "velocity_field.h"
#include <stdlib.h>
#include <string.h>

void initialize_velocity_field(VelocityField *v_field) {
    v_field->v_x = (DTYPE*) malloc(GRID_SIZE);
    v_field->v_y = (DTYPE*) malloc(GRID_SIZE);
    v_field->v_z = (DTYPE*) malloc(GRID_SIZE);

    rand_fill(v_field->v_x);
    rand_fill(v_field->v_y);
    rand_fill(v_field->v_z);
}

// TODO handle boundaries conditions

DTYPE compute_velocity_xx_grad(DTYPE *v_component, size_t i, size_t j, size_t k){
    // Gradient is done along the x-direction -> (i-1,i,i+1)
    size_t idx = rowmaj_idx(i,j,k);
    size_t left_idx = rowmaj_idx(i-1,j,k);
    size_t right_idx = rowmaj_idx(i+1,j,k);

    return (v_component[left_idx] - 2*v_component[idx] + v_component[right_idx]) * DX_INVERSE_SQUARED;
}

DTYPE compute_velocity_yy_grad(DTYPE *v_component, size_t i, size_t j, size_t k){
    // Gradient is done along the y-direction -> (j-1,j,j+1)
    size_t idx = rowmaj_idx(i,j,k);
    size_t left_idx = rowmaj_idx(i,j-1,k);
    size_t right_idx = rowmaj_idx(i,j+1,k);

    return (v_component[left_idx] - 2*v_component[idx] + v_component[right_idx]) * DY_INVERSE_SQUARED;
}

DTYPE compute_velocity_zz_grad(DTYPE *v_component, size_t i, size_t j, size_t k){
    // Gradient is done along the z-direction -> (z-1,z,z+1)
    size_t idx = rowmaj_idx(i,j,k);
    size_t left_idx = rowmaj_idx(i,j,k-1);
    size_t right_idx = rowmaj_idx(i,j,k+1);

    return (v_component[left_idx] - 2*v_component[idx] + v_component[right_idx]) * DZ_INVERSE_SQUARED;
}

void free_velocity_field(VelocityField *v_field) {
    free(v_field->v_x);
    free(v_field->v_y);
    free(v_field->v_z);
}