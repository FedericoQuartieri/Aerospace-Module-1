#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
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

// Fill an array with random values between 0 and 1
static inline void const_fill(DTYPE *component) {
    for(size_t i = 0; i < GRID_ELEMENTS; i++) {
        component[i] = 1.0;
    }
}

// Return 1 if the cell is at the boundary, 0 otherwise
// using row major representation: i = width, j = height, k = depth
static inline int is_boundary(size_t i, size_t j, size_t k) {
    return (i == 0 || i == WIDTH  - 1 ||
            j == 0 || j == HEIGHT - 1 ||
            k == 0 || k == DEPTH  - 1);
}

static inline DTYPE compute_beta_from_gamma(DTYPE gamma){
    return (DT * NU) / (2.0 * gamma);
}

static inline DTYPE compute_k_from_beta(DTYPE beta){
    return (DT * NU) / (2.0 * (beta - 1.0));
}

static inline uint64_t time_ns(void) {
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

#endif // UTILS_H