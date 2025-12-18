#include "velocity_field.h"
#include <stdlib.h>
#include <string.h>

// da capire come dare in input una funzione per il boundary
void initialize_velocity_field(VelocityField *v_field, VelocityInitFunc vx_boundary, VelocityInitFunc vy_boundary, VelocityInitFunc vz_boundary) {
    v_field->v_x = (DTYPE*) malloc(GRID_SIZE);
    v_field->v_y = (DTYPE*) malloc(GRID_SIZE);
    v_field->v_z = (DTYPE*) malloc(GRID_SIZE);

    rand_fill(v_field->v_x);
    rand_fill(v_field->v_y);
    rand_fill(v_field->v_z);

    size_t idx;

    // (i,j,k) = (0,j,k)
    for(int k = 1; k < DEPTH; k++){
        for(int j = 1; j < HEIGHT; j++){
            idx = rowmaj_idx(0,j,k);
            v_field->v_x[idx] = vx_boundary(0,j,k) - ((vy_boundary(0,j,k) - vy_boundary(0,j-1,k)) * DY_INVERSE) - ((vz_boundary(0,j,k) - vz_boundary(0,j,k-1)) * DZ_INVERSE);
            v_field->v_y[idx] = vy_boundary(0,j,k);
            v_field->v_z[idx] = vz_boundary(0,j,k);
        }
    }

    // (i,j,k) = (i,0,k)
    for(int k = 1; k < DEPTH; k++){
        for(int i = 1; i < WIDTH; i++){
            idx = rowmaj_idx(i,0,k);
            v_field->v_x[idx] = vx_boundary(i,0,k);
            v_field->v_y[idx] = vy_boundary(i,0,k) - ((vx_boundary(i,0,k) - vx_boundary(i-1,0,k)) * DX_INVERSE) - ((vz_boundary(i,0,k) - vz_boundary(i,0,k-1)) * DZ_INVERSE);
            v_field->v_z[idx] = vz_boundary(i,0,k);
        }
    }

    // (i,j,k) = (i,j,0)
    for(int j = 1; j < HEIGHT; j++){
        for(int i = 1; i < WIDTH; i++){
            idx = rowmaj_idx(i,j,0);
            v_field->v_x[idx] = vx_boundary(i,j,0);
            v_field->v_y[idx] = vy_boundary(i,j,0);
            v_field->v_z[idx] = vz_boundary(i,j,0) - ((vx_boundary(i,j,0) - vx_boundary(i-1,j,0)) * DX_INVERSE) - ((vy_boundary(i,j,0) - vy_boundary(i,j-1,0)) * DY_INVERSE); 
        }
    }

    // (i,j,k) = (0,0,k)
    for(int k = 1; k < DEPTH; k++){
        idx = rowmaj_idx(0,0,k);
        v_field->v_x[idx] = vx_boundary(0,0,k) - ((vy_boundary(0,0,k) - vy_boundary(0,0,k)) * DY_INVERSE) - ((vz_boundary(0,0,k) - vz_boundary(0,0,k-1)) * DZ_INVERSE);
        v_field->v_y[idx] = vy_boundary(0,0,k) - ((vx_boundary(0,0,k) - vx_boundary(0,0,k)) * DX_INVERSE) - ((vz_boundary(0,0,k) - vz_boundary(0,0,k-1)) * DZ_INVERSE);
        v_field->v_z[idx] = vz_boundary(0,0,k);
    }

    // (i,j,k) = (0,j,0)
    for(int j = 1; j < HEIGHT; j++){
        idx = rowmaj_idx(0,j,0);
        v_field->v_x[idx] = vx_boundary(0,j,0) - ((vy_boundary(0,j,0) - vy_boundary(0,j-1,0)) * DY_INVERSE) - ((vz_boundary(0,j,0) - vz_boundary(0,j,0)) * DZ_INVERSE);
        v_field->v_y[idx] = vy_boundary(0,j,0);
        v_field->v_z[idx] = vz_boundary(0,j,0) - ((vx_boundary(0,j,0) - vx_boundary(0,j,0)) * DX_INVERSE) - ((vy_boundary(0,j,0) - vy_boundary(0,j-1,0)) * DY_INVERSE);
    }

    // (i,j,k) = (i,0,0)
    for(int i = 1; i < WIDTH; i++){
        idx = rowmaj_idx(i,0,0);
        v_field->v_x[idx] = vx_boundary(i,0,0);
        v_field->v_y[idx] = vy_boundary(i,0,0) - ((vx_boundary(i,0,0) - vx_boundary(i-1,0,0)) * DX_INVERSE) - ((vz_boundary(i,0,0) - vz_boundary(i,0,0)) * DZ_INVERSE);
        v_field->v_z[idx] = vz_boundary(i,0,0) - ((vx_boundary(i,0,0) - vx_boundary(i-1,0,0)) * DX_INVERSE) - ((vy_boundary(i,0,0) - vy_boundary(i,0,0)) * DY_INVERSE);
    }

    // (i,j,k) = (0,0,0)
    idx = rowmaj_idx(0,0,0);
    v_field->v_x[idx] = vx_boundary(0,0,0);
    v_field->v_y[idx] = vy_boundary(0,0,0);
    v_field->v_z[idx] = vz_boundary(0,0,0);
}



DTYPE compute_velocity_x_grad(DTYPE *v_component, size_t i, size_t j, size_t k){
    // Gradient is done along the x-direction -> (i-1,i+1)
    size_t idx = rowmaj_idx(i,j,k);
    size_t neighbour = rowmaj_idx(i+1,j,k); // i+1 since i'm doing x_gradient

    return (v_component[neighbour] - v_component[idx]) * DX_INVERSE;
}

DTYPE compute_velocity_y_grad(DTYPE *v_component, size_t i, size_t j, size_t k){
    size_t idx = rowmaj_idx(i,j,k);
    size_t neighbour = rowmaj_idx(i,j+1,k);

    return (v_component[neighbour] - v_component[idx]) * DY_INVERSE;
}

DTYPE compute_velocity_z_grad(DTYPE *v_component, size_t i, size_t j, size_t k){
    size_t idx = rowmaj_idx(i,j,k);
    size_t neighbour = rowmaj_idx(i,j,k+1);

    return (v_component[neighbour] - v_component[idx]) * DZ_INVERSE;
}

DTYPE compute_velocity_xx_grad(DTYPE *v_component, size_t i, size_t j, size_t k){
    // Gradient is done along the x-direction -> (i-1,i,i+1)
    size_t idx = rowmaj_idx(i,j,k);
    size_t left_idx = rowmaj_idx(i-1,j,k);
    size_t right_idx = rowmaj_idx(i+1,j,k);

    return (v_component[left_idx] - 2*v_component[idx] + v_component[right_idx]) * DX_INVERSE_SQUARE;
}

DTYPE compute_velocity_yy_grad(DTYPE *v_component, size_t i, size_t j, size_t k){
    // Gradient is done along the y-direction -> (j-1,j,j+1)
    size_t idx = rowmaj_idx(i,j,k);
    size_t left_idx = rowmaj_idx(i,j-1,k);
    size_t right_idx = rowmaj_idx(i,j+1,k);

    return (v_component[left_idx] - 2*v_component[idx] + v_component[right_idx]) * DY_INVERSE_SQUARE;
}

DTYPE compute_velocity_zz_grad(DTYPE *v_component, size_t i, size_t j, size_t k){
    // Gradient is done along the z-direction -> (z-1,z,z+1)
    size_t idx = rowmaj_idx(i,j,k);
    size_t left_idx = rowmaj_idx(i,j,k-1);
    size_t right_idx = rowmaj_idx(i,j,k+1);

    return (v_component[left_idx] - 2*v_component[idx] + v_component[right_idx]) * DZ_INVERSE_SQUARE;
}

void free_velocity_field(VelocityField *v_field) {
    free(v_field->v_x);
    free(v_field->v_y);
    free(v_field->v_z);
}