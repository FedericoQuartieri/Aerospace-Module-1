
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* WARNING: multiple of tile size */
#define WIDTH 320
#define HEIGHT 320
#define DEPTH 32

#ifdef FLOAT
    #define TILE_WIDTH 8
    #define DTYPE float
#else
    #define TILE_WIDTH 4
    #define DTYPE double
#endif

#define TILE_HEIGHT 2
#define TILE_DEPTH 2

#define k_const 0.5
#define nu_const 0.7

#define TIME_IT(func_call, avg_iter)                            \
do {                                                            \
    long elapsed_ns_avg = 0;                                    \
    func_call; /* Warmup run. */                                \
    for (int i = 0; i < avg_iter; ++i) {                        \
        struct timespec start, stop;                            \
        clock_gettime(CLOCK_MONOTONIC, &start);                 \
        func_call;                                              \
        clock_gettime(CLOCK_MONOTONIC, &stop);                  \
        elapsed_ns_avg += (stop.tv_sec - start.tv_sec) * 1e9 +  \
                          (stop.tv_nsec - start.tv_nsec);       \
    }                                                           \
    printf(#func_call " [" #avg_iter " runs avg] %f ms\n",      \
           elapsed_ns_avg / (1e6 * avg_iter));                  \
} while (0)

static void rand_fill(DTYPE *dst, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        dst[i] = ((DTYPE) rand()) / RAND_MAX;
    }
}

static void zero_fill(DTYPE *dst, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        dst[i] = 0;
    }
}

static inline __attribute__((always_inline)) size_t rowmaj_idx(size_t i,
                                                               size_t j,
                                                               size_t k,
                                                               size_t height,
                                                               size_t width)
{
    size_t face_size = width * height;
    return i * face_size + j * width + k;
}


void comp_grad(const DTYPE *__restrict__ field,
               int depth,
               int height,
               int width,
               DTYPE *__restrict__ grad_i,
               DTYPE *__restrict__ grad_j,
               DTYPE *__restrict__ grad_k)
{
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < width; ++k) {

                size_t idx = rowmaj_idx(i, j, k, height, width);
                DTYPE value = field[idx];

                grad_i[idx] =
                    field[rowmaj_idx(i + 1, j, k, height, width)] - value;
                grad_j[idx] =
                    field[rowmaj_idx(i, j + 1, k, height, width)] - value;
                grad_k[idx] =
                    field[rowmaj_idx(i, j, k + 1, height, width)] - value;
            }
        }
    }
}

void comp_grad_vectorized();

void comp_grad_tiled_vectorized();

void comp_g(const DTYPE *f_x,
            const DTYPE *f_y,
            const DTYPE *f_z,
            const DTYPE *eta_x,
            const DTYPE *eta_y,
            const DTYPE *eta_z,
            const DTYPE *zeta_x,
            const DTYPE *zeta_y,
            const DTYPE *zeta_z,
            const DTYPE *speed_x,
            const DTYPE *speed_y,
            const DTYPE *speed_z,
            const DTYPE *grad_x,
            const DTYPE *grad_y,
            const DTYPE *grad_z,
            DTYPE *g_x,
            DTYPE *g_y,
            DTYPE *g_z,
            int depth,
            int height,
            int width)
{
    for (int i = 1; i < depth; ++i) {
        for (int j = 1; j < height; ++j) {
            for (int k = 1; k < width; ++k) {

                size_t idx = rowmaj_idx(i, j, k, height, width);

                g_x[idx] = f_x[idx] - grad_x[idx] - ((2 * nu_const) / k_const) * speed_x[idx] +
                             (nu_const / 2) * ((eta_x[rowmaj_idx(i + 1, j, k, height, width)] -2 * eta_x[idx] +
                                                 eta_x[rowmaj_idx(i-1, j, k, height, width)]) * (zeta_x[rowmaj_idx(i + 1, j, k, height, width)] -2 * zeta_x[idx] +
                                                 zeta_x[rowmaj_idx(i-1, j, k, height, width)]) + (speed_x[rowmaj_idx(i + 1, j, k, height, width)] -2 * speed_x[idx] +
                                                 speed_x[rowmaj_idx(i-1, j, k, height, width)]));
                g_y[idx] = f_y[idx] - grad_y[idx] - ((2 * nu_const) / k_const) * speed_y[idx] *
                             (nu_const / 2) * ((eta_y[rowmaj_idx(i + 1, j, k, height, width)] -2 * eta_y[idx] +
                                                 eta_y[rowmaj_idx(i-1, j, k, height, width)]) * (zeta_y[rowmaj_idx(i + 1, j, k, height, width)] -2 * zeta_y[idx] +
                                                 zeta_y[rowmaj_idx(i-1, j, k, height, width)]) + (speed_y[rowmaj_idx(i + 1, j, k, height, width)] -2 * speed_y[idx] +
                                                 speed_y[rowmaj_idx(i-1, j, k, height, width)]));
                g_z[idx] = f_z[idx] - grad_z[idx] - ((2 * nu_const) / k_const) * speed_z[idx] *
                             (nu_const / 2) * ((eta_z[rowmaj_idx(i + 1, j, k, height, width)] -2 * eta_z[idx] +
                                                 eta_z[rowmaj_idx(i-1, j, k, height, width)]) * (zeta_z[rowmaj_idx(i + 1, j, k, height, width)] -2 * zeta_z[idx] +
                                                 zeta_z[rowmaj_idx(i-1, j, k, height, width)]) + (speed_z[rowmaj_idx(i + 1, j, k, height, width)] -2 * speed_z[idx] +
                                                 speed_z[rowmaj_idx(i-1, j, k, height, width)]));
            }
        }
    }
}

int main(void)
{
    /* +1 for ghost cells. */
    const size_t size = (WIDTH + 1) * (HEIGHT + 1) * (DEPTH + 1) * sizeof(DTYPE);

    DTYPE *field = malloc(size);
    DTYPE *field_tiled = malloc(size);
    DTYPE *grad_x = malloc(size);
    DTYPE *grad_y = malloc(size);
    DTYPE *grad_z = malloc(size);


    //
    DTYPE *g_x = malloc(size);
    DTYPE *g_y = malloc(size);
    DTYPE *g_z = malloc(size);
    DTYPE *f_x = malloc(size);
    DTYPE *f_y = malloc(size);
    DTYPE *f_z = malloc(size);
    DTYPE *eta_x = malloc(size);
    DTYPE *eta_y = malloc(size);
    DTYPE *eta_z = malloc(size);
    DTYPE *zeta_x = malloc(size);
    DTYPE *zeta_y = malloc(size);
    DTYPE *zeta_z = malloc(size);
    DTYPE *speed_x = malloc(size);
    DTYPE *speed_y = malloc(size);
    DTYPE *speed_z = malloc(size);

    memset(g_x, 0, size);
    memset(g_y, 0, size);
    memset(g_z, 0, size);
    memset(f_x, 0, size);
    memset(f_y, 0, size);
    memset(f_z, 0, size);
    memset(eta_x, 0, size);
    memset(eta_y, 0, size);
    memset(eta_z, 0, size);
    memset(zeta_x, 0, size);
    memset(zeta_y, 0, size);
    memset(zeta_z, 0, size);
    memset(speed_x, 0, size);
    memset(speed_y, 0, size);
    memset(speed_z, 0, size);
    //

    memset(field, 0, size);
    memset(field_tiled, 0, size);
    memset(grad_x, 0, size);
    memset(grad_y, 0, size);
    memset(grad_z, 0, size);

    rand_fill(field, WIDTH * HEIGHT * DEPTH);
    rand_fill(f_x, WIDTH * HEIGHT * DEPTH);
    rand_fill(f_y, WIDTH * HEIGHT * DEPTH);
    rand_fill(f_z, WIDTH * HEIGHT * DEPTH);
    rand_fill(eta_x, WIDTH * HEIGHT * DEPTH);
    rand_fill(eta_y, WIDTH * HEIGHT * DEPTH);
    rand_fill(eta_z, WIDTH * HEIGHT * DEPTH);
    rand_fill(zeta_x, WIDTH * HEIGHT * DEPTH);
    rand_fill(zeta_y, WIDTH * HEIGHT * DEPTH);
    rand_fill(zeta_z, WIDTH * HEIGHT * DEPTH);
    rand_fill(speed_x, WIDTH * HEIGHT * DEPTH);
    rand_fill(speed_y, WIDTH * HEIGHT * DEPTH);
    rand_fill(speed_z, WIDTH * HEIGHT * DEPTH);
    zero_fill(g_x, WIDTH * HEIGHT * DEPTH);
    zero_fill(g_y, WIDTH * HEIGHT * DEPTH);
    zero_fill(g_z, WIDTH * HEIGHT * DEPTH);

    //
    comp_g(f_x, f_y, f_z,
           eta_x, eta_y, eta_z,
           zeta_x, zeta_y, zeta_z,
           speed_x, speed_y, speed_z,
           grad_x, grad_y, grad_z,
           g_x, g_y, g_z,
           DEPTH, HEIGHT, WIDTH);
    //

    rowmaj_to_tiled(field,
                    DEPTH,
                    HEIGHT,
                    WIDTH,
                    TILE_DEPTH,
                    TILE_HEIGHT,
                    TILE_WIDTH,
                    field_tiled);

    TIME_IT(comp_grad(field, DEPTH, HEIGHT, WIDTH, grad_z, grad_y, grad_x), 5);


    free(grad_z);
    free(grad_y);
    free(grad_x);

    free(field_tiled);
    free(field);

    return 0;
}
