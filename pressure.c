#include "pressure.h"
#include <stdlib.h>
#include <string.h>

void initialize_pressure(Pressure *pressure){
    pressure->p = (DTYPE *) malloc(GRID_SIZE);
    rand_fill(pressure->p);
}

// TODO handle boundaries conditions

DTYPE compute_pressure_x_grad(DTYPE *p, size_t i, size_t j, size_t k){
    size_t idx = rowmaj_idx(i,j,k);
    size_t neighbour = rowmaj_idx(i+1,j,k); // i+1 since i'm doing x_gradient

    return (p[neighbour] - p[idx]) / DX;
}

DTYPE compute_pressure_y_grad(DTYPE *p, size_t i, size_t j, size_t k){
    size_t idx = rowmaj_idx(i,j,k);
    size_t neighbour = rowmaj_idx(i,j+1,k); // j+1 since i'm doing y_gradient

    return (p[neighbour] - p[idx]) / DY;
}

DTYPE compute_pressure_z_grad(DTYPE *p, size_t i, size_t j, size_t k){
    size_t idx = rowmaj_idx(i,j,k);
    size_t neighbour = rowmaj_idx(i,j,k+1); // k+1 since i'm doing z_gradient

    return (p[neighbour] - p[idx]) / DZ;
}

void free_pressure(Pressure *pressure){
    free(pressure->p);
}



