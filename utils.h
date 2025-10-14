#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "constants.h"

// Compute the row-major index for a 3D grid
inline size_t rowmaj_idx(size_t i, size_t j, size_t k) {
    return i * (WIDTH * HEIGHT) + j * WIDTH + k;
}

// Fill an array with random values between 0 and 1
inline void rand_fill(DTYPE *component) {
    for(size_t i = 0; i < GRID_ELEMENTS; i++) {
        component[i] = ((DTYPE) rand()) / RAND_MAX;
    }
}

#endif // UTILS_H