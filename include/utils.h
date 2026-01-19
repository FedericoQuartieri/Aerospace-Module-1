#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "../include/constants.h"

// Compute the row-major index for a 3D grid, with: i = width, j = height, k = depth
static inline size_t rowmaj_idx(size_t i, size_t j, size_t k) {
    return k * (WIDTH * HEIGHT) + j * WIDTH + i;
}

// Fill an array with random values between 0 and 1
static inline void rand_fill(DTYPE *component) {
    for(size_t i = 0; i < GRID_ELEMENTS; i++) {
        component[i] = ((DTYPE) rand()) / RAND_MAX;
    }
}

// Return 1 if the cell is at the boundary, 0 otherwise
// using row major representation: i = width, j = height, k = depth
static inline int is_boundary(size_t i, size_t j, size_t k) {
    return (i == 0 || i == WIDTH  - 1 ||
            j == 0 || j == HEIGHT - 1 ||
            k == 0 || k == DEPTH  - 1);
}

#endif // UTILS_H