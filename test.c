#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "lin-solver.h"
#include "ftype.h"

#define SEED 42
#define SUCCESS 0
#define FAILURE 1

#define EXPECT_SUCCESS(test_call)                           \
do {                                                        \
    if (test_call == SUCCESS) {                             \
        printf("[SUCCESS] " #test_call "\n");               \
    } else {                                                \
        printf("[FAILURE] " #test_call "\n");               \
    }                                                       \
} while (0)

static ftype abs_(ftype x) { return (x >= 0) ? x : -x; }

static int verify_wD_solution(const ftype *__restrict__ w,
                              const ftype *__restrict__ f,
                              const ftype *__restrict__ u,
                              uint32_t depth,
                              uint32_t height,
                              uint32_t width,
                              uint32_t stride_i,
                              uint32_t stride_j,
                              uint32_t stride_k,
                              ftype tol)
{
    for (uint32_t i = 0; i < depth; ++i) {
        for (uint32_t j = 0; j < height; ++j) {
            ftype error = 0.0f;
            uint64_t idx = stride_i * i + stride_j * j;
            /* Check first row of the block. */
            ftype w_i = w[idx];
            ftype f_i = f[idx];
            error += abs_(f_i - ((1 + 2 * w_i) * u[idx] -
                                  w_i * u[idx + stride_k]));
            /* Check remaining rows except last one. */
            for (uint32_t k = 1; k < width - 1; ++k) {
                ftype w_i = w[idx + stride_k * k];
                ftype f_i = f[idx + stride_k * k];
                error += abs_(f_i -
                              (-w_i * u[idx + stride_k * (k - 1)] +
                               (1 + 2 * w_i) * u[idx + stride_k * k] -
                               w_i * u[idx + stride_k * (k + 1)]));
            }
            /* Check last row of the block. */
            w_i = w[idx + stride_k * (width - 1)];
            f_i = f[idx + stride_k * (width - 1)];
            error += abs_(f_i -
                          (-w_i * u[idx + stride_k * (width - 2)] +
                           (1 + 2 * w_i) *
                           u[idx + stride_k * (width - 1)]));

            if (error > tol) {
                return FAILURE;
            }
        }
    }
    return SUCCESS;
}

static int verify_wDxx_solution(const ftype *__restrict__ w,
                                const ftype *__restrict__ f,
                                const ftype *__restrict__ u,
                                uint32_t depth,
                                uint32_t height,
                                uint32_t width,
                                ftype tol)
{
    return verify_wD_solution(
        w, f, u, depth, height, width, height * width, width, 1, tol);
}

static int verify_wDyy_solution(const ftype *__restrict__ w,
                                const ftype *__restrict__ f,
                                const ftype *__restrict__ u,
                                uint32_t depth,
                                uint32_t height,
                                uint32_t width,
                                ftype tol)
{
    return verify_wD_solution(
        w, f, u, depth, width, height, height * width, 1, width, tol);
}

static int verify_wDzz_solution(const ftype *__restrict__ w,
                                const ftype *__restrict__ f,
                                const ftype *__restrict__ u,
                                uint32_t depth,
                                uint32_t height,
                                uint32_t width,
                                ftype tol)
{
    return verify_wD_solution(
        w, f, u, height, width, depth, width, 1, height * width, tol);
}


#define DEFINE_TEST_WD_SOLVER(axes) \
int test_wD##axes##_solver(uint32_t depth,                               \
                           uint32_t height,                              \
                           uint32_t width)                               \
{                                                                        \
    size_t size = (depth + VLEN) * (height + VLEN) * (width + VLEN);     \
    ftype *w = aligned_alloc(32, size * sizeof(ftype));                  \
    ftype *u = aligned_alloc(32, size * sizeof(ftype));                  \
    ftype *f = aligned_alloc(32, size * sizeof(ftype));                  \
    ftype *f_cp = aligned_alloc(32, size * sizeof(ftype));               \
    ftype *tmp = aligned_alloc(32, size * sizeof(ftype));                \
                                                                         \
    for (int i = 0; i < depth * height * width; ++i) {                   \
        w[i] = ((ftype) rand()) / RAND_MAX;                              \
        f[i] = ((ftype) rand()) / RAND_MAX;                              \
        f_cp[i] = f[i];                                                  \
    }                                                                    \
                                                                         \
    solve_wD##axes##_tridiag_blocks(w, depth, height, width, tmp, f, u); \
                                                                         \
    int status = verify_wD##axes##_solution(                             \
        w, f_cp, u, depth, height, width, 1e-4);                         \
                                                                         \
    free(tmp);                                                           \
    free(f_cp);                                                          \
    free(f);                                                             \
    free(u);                                                             \
    free(w);                                                             \
                                                                         \
    return status;                                                       \
}                                                                        \

DEFINE_TEST_WD_SOLVER(xx)
DEFINE_TEST_WD_SOLVER(yy)
DEFINE_TEST_WD_SOLVER(zz)

int main(void)
{
    srand(SEED);

    EXPECT_SUCCESS(test_wDxx_solver(1, 16, 16));
    EXPECT_SUCCESS(test_wDxx_solver(128, 128, 64));
    EXPECT_SUCCESS(test_wDxx_solver(512, 32, 256));

    EXPECT_SUCCESS(test_wDyy_solver(32, 64, 64));
    EXPECT_SUCCESS(test_wDyy_solver(128, 128, 64));
    EXPECT_SUCCESS(test_wDyy_solver(512, 32, 256));

    EXPECT_SUCCESS(test_wDzz_solver(32, 64, 64));
    EXPECT_SUCCESS(test_wDzz_solver(128, 128, 64));
    EXPECT_SUCCESS(test_wDzz_solver(32, 32, 256));

    return 0;
}
