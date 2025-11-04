#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>
#include "constants.h"
/* 
    Test optimization for computing the gradient of the pressure in row majer representation 

    compiler flags used to compare manual optimizations with compiler ones:
    flags comp: -O3 -march=native -ffast-math -funroll-loops -Xpreprocessor -fopenmp
*/

double macOS_wall_time_sec() {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0)
        mach_timebase_info(&info);

    uint64_t t = mach_absolute_time();
    return (double)t * (double)info.numer / (double)info.denom / 1e9;
}

static inline int rowmaj(int i, int j, int k){
    int layer = N * N;
    return k * layer + j * N + i;
}

// fill a matrix 512x512x512 with random values between 0 and 1
static inline void rand_fill(TYPE *p){
    for(int i = 0; i < GRID_ELEMENTS; i++){
        p[i] = ((TYPE) rand()) / RAND_MAX;
    }
}
/* -------------------------------------------------------------------------------------- */
/* 
    In the naive solution, the access pattern for the x direction is the best case for cache usage, 
    where for the y,z direction we have stride access that leads to cache misses and bad reuse of cache lines.

    However this can be well optimized by the compiler with optimization flags
*/

void comp_grad(TYPE *p, TYPE *g_x, TYPE *g_y, TYPE *g_z) {
    for(int k = 0; k < N; k++){
        for(int j = 0; j < N; j++){
            for(int i = 0; i < N; i++){
                int idx = rowmaj(i,j,k);
                TYPE p_idx = p[idx];

                int neigh = rowmaj(i+1,j,k);
                g_x[idx] = p[neigh] - p_idx;
                
                neigh = rowmaj(i,j+1,k);
                g_y[idx] = p[neigh] - p_idx;

                neigh = rowmaj(i,j,k+1);
                g_z[idx] = p[neigh] - p_idx;
            }
        }
    }
}

void comp_grad_x(TYPE *p, TYPE *g) {
    for(int k = 0; k < N; k++){
        for(int j = 0; j < N; j++){
            for(int i = 0; i < N-1; i++){
                int idx = rowmaj(i,j,k);
                int neigh = rowmaj(i+1,j,k);
                g[idx] = p[neigh] - p[idx];
            }
        }
    }
}

void comp_grad_y(TYPE *p, TYPE *g) {
    for(int k = 0; k < N; k++){
        for(int j = 0; j < N-1; j++){
            for(int i = 0; i < N; i++){
                int idx = rowmaj(i,j,k);
                int neigh = rowmaj(i,j+1,k);
                g[idx] = p[neigh] - p[idx];
            }
        }
    }
}
void comp_grad_z(TYPE *p, TYPE *g) {
    for(int k = 0; k < N-1; k++){
        for(int j = 0; j < N; j++){
            for(int i = 0; i < N; i++){
                int idx = rowmaj(i,j,k);
                int neigh = rowmaj(i,j,k+1);
                g[idx] = p[neigh] - p[idx];
            }
        }
    }
}
/* -------------------------------------------------------------------------------------- */
/* 
    Optimizations:

    - Tiling:
    divide the grid in tiles of size TxTxT
    for each tile, load the data in cache, compute the gradient for the inner cells of the tile
    then write back the results to main memory, this will reduce the number of memory accesses and cache misses,
    improving the cache reuse.

    - Vectorization:
    the memory is organized in row major, so i'm accessing through rows in the same face, then moving to the next face
    the main optimization is vectorize operations on rows:
    y_direction: for each face, compute gradient with vectorized operations on rows
    z_direction: for each row, compute gradient with vectorized operations on faces
    for the x direction it's more difficult, because I can't use the cache well since I can't access contiguous memory locations
*/

/* Tiling version to improve cache access and reuse */

#define TILE_SIZE 32
#define TILE_BLOCK TILE_SIZE * TILE_SIZE * TILE_SIZE
/* 
    dimension of the tile block: 
    - float: TILE_SIZE^3 * 4 bytes (32*32*32*4 = 32768 bytes = 32 KB)
    - double: TILE_SIZE^3 * 8 bytes (32*32*32*8 = 65536 bytes = 64 KB)
*/
void comp_grad_tiled(TYPE *p, TYPE *g_x, TYPE *g_y, TYPE *g_z) {
   
    /* N is multiple of tile_block to simplify code */

   for(size_t kk = 0; kk < N; kk+=TILE_SIZE){
       for(size_t jj = 0; jj < N; jj+=TILE_SIZE){
           for(size_t ii = 0; ii < N; ii+=TILE_SIZE){

                // Compute grad inside the block to access cache friendly
                for(size_t k_tile = kk; k_tile < kk + TILE_SIZE; k_tile++){
                    for(size_t j_tile = jj; j_tile < jj + TILE_SIZE; j_tile++){
                        for(size_t i_tile = ii; i_tile < ii + TILE_SIZE; i_tile++){
                            int idx = rowmaj(i_tile,j_tile,k_tile);
                            TYPE p_idx = p[idx];

                            int neigh = rowmaj(i_tile+1,j_tile,k_tile);
                            g_x[idx] = p[neigh] - p_idx;
                            
                            neigh = rowmaj(i_tile,j_tile+1,k_tile);
                            g_y[idx] = p[neigh] - p_idx;

                            neigh = rowmaj(i_tile,j_tile,k_tile+1);
                            g_z[idx] = p[neigh] - p_idx;
                        }
                    }
                }

            }
        }
    }
}

/* -------------------------------------------------------------------------------------- */
/* this function takes a pointer to the function that will be benchmarked, the pressure field and the output gradients */
void benchmark_single(void (*func)(TYPE*, TYPE*), TYPE *pressure, TYPE *grad) {
    int iterations = 5;
    func(pressure, grad); // warm-up

    double start_ = macOS_wall_time_sec();
    for (int i = 0; i < iterations; i++) func(pressure, grad);
    double end_ = macOS_wall_time_sec();

    double time = (end_ - start_) / iterations;
    printf("\nBenchmark: %.6f s", time);

    volatile TYPE checksum = 0;
    for (int i = 0; i < GRID_ELEMENTS; i++) checksum += grad[i];
    printf("\nChecksum: %f\n", (double)checksum);
}

void benchmark(void (*func)(TYPE*, TYPE*, TYPE*, TYPE*), TYPE *pressure, TYPE *grad_x, TYPE *grad_y, TYPE *grad_z) {
    int iterations = 5;
    func(pressure, grad_x, grad_y, grad_z); // warm-up

    double start_ = macOS_wall_time_sec();
    for (int i = 0; i < iterations; i++) func(pressure, grad_x, grad_y, grad_z);
    double end_ = macOS_wall_time_sec();

    double time = (end_ - start_) / iterations;
    printf("\nBenchmark: %.6f s", time);

    volatile TYPE checksum = 0;
    for (int i = 0; i < GRID_ELEMENTS; i++) checksum += grad_x[i] + grad_y[i] + grad_z[i];
    printf("\nChecksum: %f\n", (double)checksum);
}

int main(void) {
/* ------------------------- INITIALIZATION ----------------------------- */
    TYPE *pressure = malloc(GRID_ELEMENTS * sizeof(TYPE));
    TYPE *grad_x = malloc(GRID_ELEMENTS * sizeof(TYPE));
    TYPE *grad_y = malloc(GRID_ELEMENTS * sizeof(TYPE));
    TYPE *grad_z = malloc(GRID_ELEMENTS * sizeof(TYPE)); 
    TYPE *grad_x_tiled = malloc(GRID_ELEMENTS * sizeof(TYPE));
    TYPE *grad_y_tiled = malloc(GRID_ELEMENTS * sizeof(TYPE));
    TYPE *grad_z_tiled = malloc(GRID_ELEMENTS * sizeof(TYPE));   
    rand_fill(pressure);
    memset(grad_x, 0, GRID_ELEMENTS * sizeof(TYPE));
    memset(grad_y, 0, GRID_ELEMENTS * sizeof(TYPE));
    memset(grad_z, 0, GRID_ELEMENTS * sizeof(TYPE));
    memset(grad_x_tiled, 0, GRID_ELEMENTS * sizeof(TYPE));
    memset(grad_y_tiled, 0, GRID_ELEMENTS * sizeof(TYPE));
    memset(grad_z_tiled, 0, GRID_ELEMENTS * sizeof(TYPE));
/* ------------------------- BENCHMARKING NAIVE ----------------------------- */
    // printf("\nNaive solution x:");
    // benchmark_single(comp_grad_x, pressure, grad_x);

    // printf("\nNaive solution y:");
    // benchmark_single(comp_grad_y, pressure, grad_y);

    // printf("\nNaive solution z:");
    // benchmark_single(comp_grad_z, pressure, grad_z);

    printf("\nNaive solution full:");
    benchmark(comp_grad, pressure, grad_x, grad_y, grad_z);
/* --------------------------BENCHMARKING TILED ------------------------------ */
    printf("\nTiled solution full:");
    benchmark(comp_grad_tiled, pressure, grad_x, grad_y, grad_z);

/* --------------------------------------------------------------------------- */
    free(pressure);
    free(grad_x);
    free(grad_y);
    free(grad_z);

    printf("\n");
    return 0;
}
