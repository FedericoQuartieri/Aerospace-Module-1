#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

#include "constants.h"
#include "neon_instructions.h"

/* 
    Performance test for computing the pressure gradient 
    in a row-major 3D grid representation.

    Compiler flags used to compare manual and compiler-level optimizations:
    Example: -O3 -march=native -ffast-math -funroll-loops -Xpreprocessor -fopenmp
*/

double macOS_wall_time_sec() {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0)
        mach_timebase_info(&info);

    uint64_t t = mach_absolute_time();
    return (double)t * (double)info.numer / (double)info.denom / 1e9;
}

static inline size_t rowmaj(size_t i, size_t j, size_t k) {
    size_t layer = N * N;
    return k * layer + j * N + i;
}

// Fill a 512x512x512 grid with random values between 0 and 1
static inline void rand_fill(TYPE *p) {
    for (size_t i = 0; i < GRID_ELEMENTS; i++) {
        p[i] = ((TYPE) rand()) / RAND_MAX;
    }
}

static inline void set_grad_zero(TYPE *grad_x, TYPE *grad_y, TYPE *grad_z){
    memset(grad_x, 0, GRID_ELEMENTS * sizeof(TYPE));
    memset(grad_y, 0, GRID_ELEMENTS * sizeof(TYPE));
    memset(grad_z, 0, GRID_ELEMENTS * sizeof(TYPE));
}

/* -------------------------------------------------------------------------------------- */
/*
    In the naive implementation, the access pattern along the x-direction 
    offers the best cache locality, while accesses along the y and z directions 
    involve strided memory access, causing cache misses and poor cache line reuse.

    However, with appropriate compiler optimizations, this can be improved automatically.
*/

void comp_grad(TYPE *__restrict p, TYPE *__restrict g_x, TYPE *__restrict g_y, TYPE *__restrict g_z) {
    for (size_t k = 0; k < N-1; k++) {
        for (size_t j = 0; j < N-1; j++) {
            for (size_t i = 0; i < N-1; i++) {
                size_t idx = rowmaj(i, j, k);
                TYPE p_idx = p[idx];

                size_t neigh = rowmaj(i+1, j, k);
                g_x[idx] = p[neigh] - p_idx;
                
                neigh = rowmaj(i, j+1, k);
                g_y[idx] = p[neigh] - p_idx;

                neigh = rowmaj(i, j, k+1);
                g_z[idx] = p[neigh] - p_idx;
            }
        }
    }
}

void comp_grad_x(TYPE *__restrict p, TYPE *__restrict g) { 
    for(size_t k = 0; k < N; k++){ 
        for(size_t j = 0; j < N; j++){ 
            for(size_t i = 0; i < N-1; i++){ 
                size_t idx = rowmaj(i,j,k); 
                size_t neigh = rowmaj(i+1,j,k); 
                g[idx] = p[neigh] - p[idx]; 
            } 
        } 
    } 
} 

void comp_grad_y(TYPE *__restrict p, TYPE *__restrict g) { 
    for(size_t k = 0; k < N; k++){ 
        for(size_t j = 0; j < N-1; j++){ 
            for(size_t i = 0; i < N; i++){ 
                size_t idx = rowmaj(i,j,k); 
                size_t neigh = rowmaj(i,j+1,k); 
                g[idx] = p[neigh] - p[idx]; 
            } 
        } 
    } 
} 

void comp_grad_z(TYPE *__restrict p, TYPE *__restrict g) { 
    for(size_t k = 0; k < N-1; k++){ 
        for(size_t j = 0; j < N; j++){ 
            for(size_t i = 0; i < N; i++){ 
                size_t idx = rowmaj(i,j,k); 
                size_t neigh = rowmaj(i,j,k+1); 
                g[idx] = p[neigh] - p[idx]; 
            } 
        } 
    } 
}

/* -------------------------------------------------------------------------------------- */
/*
    Optimizations considered:

    - Tiling:
      The grid is divided into tiles of size T×T×T.
      For each tile, data are loaded into cache, the gradients for its inner cells are computed,
      and the results are written back to main memory.
      This reduces the number of memory accesses and cache misses, improving cache reuse.

    - Vectorization:
      Since the memory layout is row-major, we can perform vectorized operations along rows.
      * y-direction: for each face, compute gradients by vectorizing across rows.
      * z-direction: for each row, compute gradients by vectorizing across faces.
      For the x-direction, cache efficiency is already high since data are contiguous.

    - TLB Misses:
      Because the stencil radius is 1, memory accesses can be well-prefetched by the compiler.
      The tiled version may reduce cache misses but increase TLB misses,
      as it accesses elements from distant memory pages.
      Therefore, in this specific case, the optimized naive implementation 
      may outperform the tiled version.
*/

/* 
    TILING

    This version improves cache access and particularly cache reuse.

    In the naive implementation, cache reuse along y and z is poor
    because cells are re-accessed after they have been evicted from cache.
    Consider the y-direction:
        load p[idx]             -> cache line: p[idx], p[idx+1], p[idx+2], p[idx+3]
        load p[idx + stride]    -> cache line: p[idx+stride], p[idx+stride+1], ...
    As we move along the row, we loose cached elements before they are reused.

    In contrast, the x-direction performs all operations while the relevant
    data are still in cache, achieving better locality.

    Tiling improves performance by increasing cache reuse, ensuring that
    computations are performed while the required data remain in cache.
*/

#define TILE_SIZE 32
#define TILE_BLOCK TILE_SIZE * TILE_SIZE * TILE_SIZE

/* 
    Tile block size:
    - float:  TILE_SIZE^3 * 4 bytes  = 32 * 32 * 32 * 4  = 32 KB
    - double: TILE_SIZE^3 * 8 bytes  = 32 * 32 * 32 * 8  = 64 KB
*/
void comp_grad_tiled(TYPE *__restrict p, TYPE *__restrict g_x, TYPE *__restrict g_y, TYPE *__restrict g_z) {
   
    /* Assumes N-1 is a multiple of TILE_SIZE for simplicity */

    for (size_t kk = 0; kk < N-1; kk += TILE_SIZE) {
        for (size_t jj = 0; jj < N-1; jj += TILE_SIZE) {
            for (size_t ii = 0; ii < N-1; ii += TILE_SIZE) {

                // Compute gradients within the block
                for (size_t k_tile = kk; k_tile < kk + TILE_SIZE; k_tile++) {
                    for (size_t j_tile = jj; j_tile < jj + TILE_SIZE; j_tile++) {
                        for (size_t i_tile = ii; i_tile < ii + TILE_SIZE; i_tile++) {
                            size_t idx = rowmaj(i_tile, j_tile, k_tile);
                            TYPE p_idx = p[idx];

                            size_t neigh_x = rowmaj(i_tile+1, j_tile, k_tile);
                            g_x[idx] = p[neigh_x] - p_idx;
                            
                            size_t neigh_y = rowmaj(i_tile, j_tile+1, k_tile);
                            g_y[idx] = p[neigh_y] - p_idx;

                            size_t neigh_z = rowmaj(i_tile, j_tile, k_tile+1);
                            g_z[idx] = p[neigh_z] - p_idx;
                        }
                    }
                }
            }
        }
    }
}

/* 
    VECTORIZATION

    To further improve performance, we can vectorize computations using NEON SIMD 
    instructions (16-byte vector registers on Apple M1/M2) via <arm_neon.h>.
*/

void comp_grad_vectorized(TYPE *__restrict p, TYPE *__restrict g_x, TYPE *__restrict g_y, TYPE *__restrict g_z){
    for (size_t k = 0; k < N - 1; k++) {
        for (size_t j = 0; j < N - 1; j++) {
            size_t i = 0;
            for (; i <= N - VLEN; i += VLEN) {
                size_t idx = rowmaj(i, j, k);

                /*  x direction */
                VTYPE x = VLOAD(&p[idx]); // reuse also for y,z
                VTYPE x_1 = VLOAD(&p[idx + 1]);
                VTYPE diff = VSUB(x_1, x);
                VSTORE(&g_x[idx], diff);

                /* y direction */
                size_t neigh = rowmaj(i, j + 1, k);
                VTYPE y_1 = VLOAD(&p[neigh]);
                diff = VSUB(y_1, x);
                VSTORE(&g_y[idx], diff);

                /* z direction */
                neigh = rowmaj(i, j, k + 1);
                VTYPE z_1 = VLOAD(&p[neigh]);
                diff = VSUB(z_1, x);
                VSTORE(&g_z[idx], diff);

            }
            for (; i < N-1; i++) {
                size_t idx = rowmaj(i, j, k);

                /* x direction */
                g_x[idx] = p[idx+1] - p[idx];

                /* y direction */
                size_t neigh = rowmaj(i, j + 1, k);
                g_y[idx] = p[neigh] - p[idx];

                /* z direction */
                neigh = rowmaj(i,j,k+1);
                g_z[idx] = p[neigh] - p[idx];
            }
        }
    }
}

void comp_grad_x_vectorized(TYPE *__restrict p, TYPE *__restrict g) {
    for (size_t k = 0; k < N; k++) {
        for (size_t j = 0; j < N; j++) {
            /* In the x-direction we access contiguous memory locations */
            size_t i = 0;
            for (i = 0; i < N - VLEN - 1; i += VLEN) {
                size_t idx = rowmaj(i, j, k);
                VTYPE x = VLOAD(&p[idx]);
                VTYPE x_1 = VLOAD(&p[idx + 1]);
                VTYPE diff = VSUB(x_1, x);
                VSTORE(&g[idx], diff);
            }
            /* Handle the last elements to avoid out-of-bound access */
            for (; i < N - 1; i++) {
                size_t idx = rowmaj(i, j, k);
                g[idx] = p[idx + 1] - p[idx];
            }
        }
    }
}

void comp_grad_y_vectorized(TYPE *__restrict p, TYPE *__restrict g) {
    for (size_t k = 0; k < N; k++) {
        for (size_t j = 0; j < N - 1; j++) {
            size_t i = 0;
            for (; i <= N - VLEN; i += VLEN) {
                size_t idx = rowmaj(i, j, k);
                size_t idx_1 = rowmaj(i, j + 1, k);
                VTYPE y = VLOAD(&p[idx]);
                VTYPE y_1 = VLOAD(&p[idx_1]);
                VTYPE diff = VSUB(y_1, y);
                VSTORE(&g[idx], diff);
            }
            for (; i < N; i++) {
                size_t idx = rowmaj(i, j, k);
                size_t idx_1 = rowmaj(i, j + 1, k);
                g[idx] = p[idx_1] - p[idx];
            }
        }
    }
}

void comp_grad_z_vectorized(TYPE *__restrict p, TYPE *__restrict g) {
    
    for (size_t k = 0; k < N - 1; k++) {
        for (size_t j = 0; j < N; j++) {
            size_t i = 0;
            for (; i <= N - VLEN; i += VLEN) {
                size_t idx = rowmaj(i, j, k);
                size_t idx_1 = rowmaj(i, j, k + 1);
                VTYPE z = VLOAD(&p[idx]);
                VTYPE z_1 = VLOAD(&p[idx_1]);
                // Compute the difference between two planes
                VTYPE diff = VSUB(z_1, z);
                VSTORE(&g[idx], diff);
            }
            for (; i < N; i++) {
                size_t idx = rowmaj(i, j, k);
                size_t idx_1 = rowmaj(i, j, k + 1);
                g[idx] = p[idx_1] - p[idx];
            }
        }
    }
}

/* -------------------------------------------------------------------------------------- */
/* 
    Benchmarking utilities.
    These functions measure execution time and compute a simple checksum 
    to verify numerical consistency across implementations.
*/

void benchmark_single(void (*func)(TYPE*, TYPE*), TYPE *pressure, TYPE *grad) {
    size_t iterations = 5;
    func(pressure, grad); // warm-up

    double start_ = macOS_wall_time_sec();
    for (size_t i = 0; i < iterations; i++) func(pressure, grad);
    double end_ = macOS_wall_time_sec();

    double time = (end_ - start_) / iterations;
    printf("\nBenchmark: %.6f s", time);

    volatile TYPE checksum = 0;
    for (size_t i = 0; i < GRID_ELEMENTS; i++) checksum += grad[i];
    printf("\nChecksum: %f\n", (double)checksum);
}

void benchmark(void (*func)(TYPE*, TYPE*, TYPE*, TYPE*), TYPE *pressure, TYPE *grad_x, TYPE *grad_y, TYPE *grad_z) {
    size_t iterations = 5;
    func(pressure, grad_x, grad_y, grad_z); // warm-up

    double start_ = macOS_wall_time_sec();
    for (size_t i = 0; i < iterations; i++) func(pressure, grad_x, grad_y, grad_z);
    double end_ = macOS_wall_time_sec();

    double time = (end_ - start_) / iterations;
    printf("\nBenchmark: %.6f s", time);

    volatile TYPE checksum = 0;
    for (size_t i = 0; i < GRID_ELEMENTS; i++) checksum += grad_x[i] + grad_y[i] + grad_z[i];
    printf("\nChecksum: %f\n", (double)checksum);
}

/* -------------------------------------------------------------------------------------- */
int main(void) {
    /* ------------------------- INITIALIZATION ----------------------------- */
    TYPE *pressure = malloc(GRID_ELEMENTS * sizeof(TYPE));
    TYPE *grad_x   = malloc(GRID_ELEMENTS * sizeof(TYPE));
    TYPE *grad_y   = malloc(GRID_ELEMENTS * sizeof(TYPE));
    TYPE *grad_z   = malloc(GRID_ELEMENTS * sizeof(TYPE));   

    if (!pressure || !grad_x || !grad_y || !grad_z) {
        fprintf(stderr, "Failed to allocate %.2f GiB for %zu grid elements.\n",
                (4.0 * GRID_ELEMENTS * sizeof(TYPE)) / (1024.0 * 1024.0 * 1024.0),
                GRID_ELEMENTS);
        free(pressure);
        free(grad_x);
        free(grad_y);
        free(grad_z);
        return EXIT_FAILURE;
    }

    rand_fill(pressure);
    set_grad_zero(grad_x,grad_y,grad_z);

    /* ------------------------- NAIVE IMPLEMENTATION ----------------------------- */
/*     printf("\nNaive solution (x-direction):");
    benchmark_single(comp_grad_x, pressure, grad_x);

    printf("\nNaive solution (y-direction):");
    benchmark_single(comp_grad_y, pressure, grad_y);

    printf("\nNaive solution (z-direction):");
    benchmark_single(comp_grad_z, pressure, grad_z); 

    set_grad_zero(grad_x,grad_y,grad_z);
*/
    printf("\nNaive full-gradient solution:");
    benchmark(comp_grad, pressure, grad_x, grad_y, grad_z);

    /* -------------------------- TILED IMPLEMENTATION ------------------------------ */
    set_grad_zero(grad_x,grad_y,grad_z);

    printf("\nTiled full-gradient solution:");
    benchmark(comp_grad_tiled, pressure, grad_x, grad_y, grad_z);

    /* -------------------------- VECTORIZED IMPLEMENTATION ------------------------------ */
    set_grad_zero(grad_x,grad_y,grad_z);

/*     printf("\nVectorized solution (x-direction):");
    benchmark_single(comp_grad_x_vectorized, pressure, grad_x);

    printf("\nVectorized solution (y-direction):");
    benchmark_single(comp_grad_y_vectorized, pressure, grad_y);

    printf("\nVectorized solution (z-direction):");
    benchmark_single(comp_grad_z_vectorized, pressure, grad_z);

    set_grad_zero(grad_x,grad_y,grad_z); */

    printf("\nVectorized full-gradient solution:");
    benchmark(comp_grad_vectorized, pressure, grad_x, grad_y, grad_z);
    /* --------------------------------------------------------------------------- */
    free(pressure);
    free(grad_x);
    free(grad_y);
    free(grad_z);

    printf("\n");
    return 0;
}
