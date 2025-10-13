#include "defines.h"
#include "g.h"
#include "helpers.h"

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


int main(void)
{
    /* +1 for ghost cells. */
    const size_t size = (WIDTH + 1) * (HEIGHT + 1) * (DEPTH + 1) * sizeof(DTYPE);

    DTYPE *field = malloc(size);
    DTYPE *grad_x = malloc(size);
    DTYPE *grad_y = malloc(size);
    DTYPE *grad_z = malloc(size);
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
    DTYPE *k_values = malloc(size);

    memset(field, 0, size);
    memset(grad_x, 0, size);
    memset(grad_y, 0, size);
    memset(grad_z, 0, size);
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
    memset(k_values, 0, size);
    

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
    rand_fill(k_values, WIDTH * HEIGHT * DEPTH);

    TIME_IT(comp_grad(field, DEPTH, HEIGHT, WIDTH, grad_z, grad_y, grad_x), 5);

    //
    g(f_x, f_y, f_z,
           eta_x, eta_y, eta_z,
           zeta_x, zeta_y, zeta_z,
           speed_x, speed_y, speed_z,
           grad_x, grad_y, grad_z,
           g_x, g_y, g_z,
           k_values,
           DEPTH, HEIGHT, WIDTH);


    free(grad_z);
    free(grad_y);
    free(grad_x);
    free(field);
    free(g_x);
    free(g_y);
    free(g_z);
    free(f_x);
    free(f_y);
    free(f_z);
    free(eta_x);
    free(eta_y);
    free(eta_z);
    free(zeta_x);
    free(zeta_y);
    free(zeta_z);
    free(speed_x);
    free(speed_y);
    free(speed_z);
    free(k_values);

    return 0;
}
