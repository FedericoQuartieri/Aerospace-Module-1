#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "lin-solver.h"
#include "equations.h"
#include "boundary.h"
#include "ftype.h"
#include "utils.h"

#define SEED 42
#define SUCCESS 0
#define FAILURE 1

#define TOL 1e-4

#define ASSERT_TRUE(condition)                              \
do {                                                        \
    if (!(condition)) {                                     \
        printf("[ASSERT_TRUE FAILED] %s\n", #condition);    \
        return FAILURE;                                     \
    }                                                       \
} while (0)

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
                              ftype u0,
                              ftype un,
                              ftype tol,
                              int is_comp_normal,
                              int is_bc_dirichlet)
{
    for (uint32_t i = 0; i < depth; ++i) {
        for (uint32_t j = 0; j < height; ++j) {
            ftype error = 0.0f;
            uint64_t idx = stride_i * i + stride_j * j;
            /* Check first row of the block. */
            ftype w_i = w[idx];
            ftype f_i = f[idx];
            if (is_bc_dirichlet) {
                /* Dirichlet boundary condition! */
                error += abs_(u0 - u[idx]);
            } else {
                /* Homogeneous Neumann boundary condition! */
                error += abs_(f_i /* -2 dx u0 */ -
                              (3 * u[idx] - 2 * u[idx + stride_k]));
            }
            //printf("%2d %2d  %f - f=%f\n", i, j, error, f_i);
            /* Check remaining rows except last one. */
            for (uint32_t k = 1; k < width - 1; ++k) {
                ftype w_i = w[idx + stride_k * k];
                ftype f_i = f[idx + stride_k * k];
                ftype err = abs_(f_i /* -2 dx u0 */ -
                              (-w_i * u[idx + stride_k * (k - 1)] +
                               (1 + 2 * w_i) * u[idx + stride_k * k] -
                               w_i * u[idx + stride_k * (k + 1)]));
                error += err;

                //printf("%2d %2d  %f - f=%f u=%f\n", i, j, err, f_i, u[idx + stride_k * k]);
            }
            /* Check last row of the block. */
            w_i = w[idx + stride_k * (width - 1)];
            f_i = f[idx + stride_k * (width - 1)];
            /* Dirichlet boundary condition! */
            if (is_bc_dirichlet) {
                if (is_comp_normal) {
                    error += abs_(un - u[idx + stride_k * (width - 1)]);
                } else {
                    error += abs_((-2 * w_i * un - f_i) -
                                  ((-w_i * u[idx + stride_k * (width - 2)] +
                                   (1 + 3 * w_i) *
                                   u[idx + stride_k * (width - 1)])));
                }
            } else {
                error += abs_(f_i /* + dx un */ -
                              (-1 * u[idx + stride_k * (width - 2)] +
                               2 * u[idx + stride_k * (width - 1)]));
            }

            //printf("%2d %2d %f\n", i, j, error);

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
                                ftype u0,
                                ftype un,
                                uint32_t depth,
                                uint32_t height,
                                uint32_t width,
                                ftype tol,
                                int is_comp_normal,
                                int is_bc_dirichlet)
{
    return verify_wD_solution(w, f, u, depth, height, width,
                              height * width, width, 1, u0, un,
                              tol, is_comp_normal, is_bc_dirichlet);
}

static int verify_wDyy_solution(const ftype *__restrict__ w,
                                const ftype *__restrict__ f,
                                const ftype *__restrict__ u,
                                ftype u0,
                                ftype un,
                                uint32_t depth,
                                uint32_t height,
                                uint32_t width,
                                ftype tol,
                                int is_comp_normal,
                                int is_bc_dirichlet)
{
    return verify_wD_solution(w, f, u, depth, width, height,
                              height * width, 1, width, u0, un,
                              tol, is_comp_normal, is_bc_dirichlet);
}

static int verify_wDzz_solution(const ftype *__restrict__ w,
                                const ftype *__restrict__ f,
                                const ftype *__restrict__ u,
                                ftype u0,
                                ftype un,
                                uint32_t depth,
                                uint32_t height,
                                uint32_t width,
                                ftype tol,
                                int is_comp_normal,
                                int is_bc_dirichlet)
{
    return verify_wD_solution(w, f, u, width, height, depth,
                              1, width, height * width, u0, un,
                              tol, is_comp_normal, is_bc_dirichlet);
}

#define DEFINE_TEST_MOMENTUM_D_SOLVER(                                   \
    axes, is_x_normal, is_y_normal, is_z_normal,                         \
    u0_x, u0_y, u0_z, un_x, un_y, un_z)                                  \
int test_momentum_D##axes##_solver(uint32_t depth,                       \
                                   uint32_t height,                      \
                                   uint32_t width)                       \
{                                                                        \
    size_t size = depth * height * width;                                \
    ftype *w = aligned_alloc(32, size * sizeof(ftype));                  \
    ftype *u = aligned_alloc(32, 3 * size * sizeof(ftype));              \
    ftype *f = aligned_alloc(32, 3 * size * sizeof(ftype));              \
    ftype *f_cp = aligned_alloc(32, 3 * size * sizeof(ftype));           \
    /* WARNING: size may be smaller than 4 * width * VLEN */             \
    ftype *tmp = aligned_alloc(32, 3 * size * sizeof(ftype));            \
                                                                         \
    ftype *u_x = u;                                                      \
    ftype *f_x = f;                                                      \
    ftype *f_x_cp = f_cp;                                                \
    ftype *u_y = u + size;                                               \
    ftype *f_y = f + size;                                               \
    ftype *f_y_cp = f_cp + size;                                         \
    ftype *u_z = u + 2 * size;                                           \
    ftype *f_z = f + 2 * size;                                           \
    ftype *f_z_cp = f_cp + 2 * size;                                     \
                                                                         \
    for (int i = 0; i < depth * height * width; ++i) {                   \
        w[i] = ((ftype) rand()) / RAND_MAX;                              \
        f_x[i] = ((ftype) rand()) / RAND_MAX;                            \
        f_y[i] = ((ftype) rand()) / RAND_MAX;                            \
        f_z[i] = ((ftype) rand()) / RAND_MAX;                            \
        f_x_cp[i] = f_x[i];                                              \
        f_y_cp[i] = f_y[i];                                              \
        f_z_cp[i] = f_z[i];                                              \
    }                                                                    \
                                                                         \
    solve_momentum_D##axes(w, depth, height, width,                      \
                           tmp, f_x, f_y, f_z, u_x, u_y, u_z);           \
                                                                         \
    int error = verify_wD##axes##_solution(w, f_x_cp, u_x,               \
                                           u0_x, un_x, depth,            \
                                           height, width, TOL,           \
                                           is_x_normal, 1) ||            \
                verify_wD##axes##_solution(w, f_y_cp, u_y,               \
                                           u0_y, un_y, depth,            \
                                           height, width, TOL,           \
                                           is_y_normal, 1) ||            \
                verify_wD##axes##_solution(w, f_z_cp, u_z,               \
                                           u0_z, un_z, depth,            \
                                           height, width, TOL,           \
                                           is_z_normal, 1);              \
                                                                         \
    free(tmp);                                                           \
    free(f_cp);                                                          \
    free(f);                                                             \
    free(u);                                                             \
    free(w);                                                             \
                                                                         \
    return error;                                                        \
}                                                                        \

void compute_div(const ftype *src_x,
                 const ftype *src_y,
                 const ftype *src_z,
                 uint32_t depth,
                 uint32_t height,
                 uint32_t width,
                 ftype *dst)
{
    /* Front face divergence is zero. */
    for (uint32_t j = 0; j < height; ++j) {
        for (uint32_t k = 0; k < width; ++k) {
            dst[width * j + k] = 0;
        }
    }

    for (uint32_t i = 1; i < depth; ++i) {
        uint64_t face_offset = height * width * i;
        /* Top face divergence is zero. */
        for (int k = 0; k < width; ++k) {
            dst[face_offset + k] = 0;
        }

        for (uint32_t j = 1; j < height; ++j) {
            /* Left face divergence is zero. */
            dst[face_offset + width * j] = 0;
            for (uint32_t k = 1; k < width; ++k) {
                uint64_t idx = face_offset + width * j + k;
                dst[idx] = (src_x[idx] - src_x[idx - 1]) +
                           (src_y[idx] - src_y[idx - width]) +
                           (src_z[idx] - src_z[idx - height * width]);
            }
        }
    }
}

int test_pressure_Dxx_solver(uint32_t depth,
                             uint32_t height,
                             uint32_t width)
{
    size_t size = depth * height * width;
    size_t alloc_size = size * sizeof(ftype);
    ftype *w = aligned_alloc(32, alloc_size);
    ftype *p = aligned_alloc(32, alloc_size);
    ftype *u = aligned_alloc(32, alloc_size * 3);
    ftype *f = aligned_alloc(32, alloc_size);
    ftype *tmp = aligned_alloc(32, alloc_size);

    ftype *u_x = u;
    ftype *u_y = u_x + size;
    ftype *u_z = u_y + size;

    for (uint64_t i = 0; i < depth * height * width; ++i) {
        w[i] = 1.0;
        u_x[i] = ((ftype) rand()) / RAND_MAX;
        u_y[i] = ((ftype) rand()) / RAND_MAX;
        u_z[i] = ((ftype) rand()) / RAND_MAX;
    }

    compute_div(u_x, u_y, u_z, depth, height, width, f);

    solve_pressure_Dxx(depth, height, width, tmp, u_x, u_y, u_z, p);

    int error = verify_wDxx_solution(w, f, p,
                                     0.0, 0.0, depth,
                                     height, width, TOL, 1, 0);

    free(tmp);
    free(f);
    free(u);
    free(p);
    free(w);

    return error;
}

#define DEFINE_TEST_PRESSURE_D_SOLVER(axes)                                \
int test_pressure_D##axes##_solver(uint32_t depth,                         \
                                   uint32_t height,                        \
                                   uint32_t width)                         \
{                                                                          \
    size_t size = depth * height * width;                                  \
    size_t alloc_size = size * sizeof(ftype);                              \
    ftype *w = aligned_alloc(32, alloc_size);                              \
    ftype *p = aligned_alloc(32, alloc_size);                              \
    ftype *f = aligned_alloc(32, alloc_size);                              \
    ftype *f_cp = aligned_alloc(32, alloc_size);                           \
    /* WARNING: Only max(d, h, w) tmp space would be required. */          \
    ftype *tmp = aligned_alloc(32, alloc_size);                            \
                                                                           \
    for (uint64_t i = 0; i < depth * height * width; ++i) {                \
        w[i] = 1.0;                                                        \
        f[i] = ((ftype) rand()) / RAND_MAX;                                \
        f_cp[i] = f[i];                                                    \
    }                                                                      \
                                                                           \
    solve_pressure_D##axes(depth, height, width, tmp, f, p);               \
                                                                           \
    int error = verify_wD##axes##_solution(w, f_cp, p,                     \
                                           0.0, 0.0, depth,                \
                                           height, width, TOL, 1, 0);      \
                                                                           \
    free(tmp);                                                             \
    free(f);                                                               \
    free(p);                                                               \
    free(w);                                                               \
                                                                           \
    return error;                                                          \
}

DEFINE_TEST_PRESSURE_D_SOLVER(yy)
DEFINE_TEST_PRESSURE_D_SOLVER(zz)

#define LEFT_BC_U_X 0.1
#define LEFT_BC_U_Y 0.2
#define LEFT_BC_U_Z 0.3
#define RIGHT_BC_U_X 0.4
#define RIGHT_BC_U_Y 0.5
#define RIGHT_BC_U_Z 0.6
#define TOP_BC_U_X -0.1
#define TOP_BC_U_Y -0.2
#define TOP_BC_U_Z -0.3
#define BOTTOM_BC_U_X -0.3
#define BOTTOM_BC_U_Y -0.5
#define BOTTOM_BC_U_Z -0.6
#define FRONT_BC_U_X 0.7
#define FRONT_BC_U_Y 0.9
#define FRONT_BC_U_Z 0.1
#define BACK_BC_U_X 0.0
#define BACK_BC_U_Y -0.3
#define BACK_BC_U_Z 0.8

DEFINE_CONSTANT_BC_U(LEFT_BC_U_X, LEFT_BC_U_Y, LEFT_BC_U_Z, BC_LEFT)
DEFINE_CONSTANT_BC_U(RIGHT_BC_U_X, RIGHT_BC_U_Y, RIGHT_BC_U_Z, BC_RIGHT)
DEFINE_CONSTANT_BC_U(TOP_BC_U_X, TOP_BC_U_Y, TOP_BC_U_Z, BC_TOP)
DEFINE_CONSTANT_BC_U(BOTTOM_BC_U_X, BOTTOM_BC_U_Y, BOTTOM_BC_U_Z, BC_BOTTOM)
DEFINE_CONSTANT_BC_U(FRONT_BC_U_X, FRONT_BC_U_Y, FRONT_BC_U_Z, BC_FRONT)
DEFINE_CONSTANT_BC_U(BACK_BC_U_X, BACK_BC_U_Y, BACK_BC_U_Z, BC_BACK)

DEFINE_TEST_MOMENTUM_D_SOLVER(
    xx, 1, 0, 0, LEFT_BC_U_X, LEFT_BC_U_Y, LEFT_BC_U_Z,
    RIGHT_BC_U_X, RIGHT_BC_U_Y, RIGHT_BC_U_Z)

DEFINE_TEST_MOMENTUM_D_SOLVER(
    yy, 0, 1, 0, TOP_BC_U_X, TOP_BC_U_Y, TOP_BC_U_Z,
    BOTTOM_BC_U_X, BOTTOM_BC_U_Y, BOTTOM_BC_U_Z)

DEFINE_TEST_MOMENTUM_D_SOLVER(
    zz, 0, 0, 1, FRONT_BC_U_X, FRONT_BC_U_Y, FRONT_BC_U_Z,
    BACK_BC_U_X, BACK_BC_U_Y, BACK_BC_U_Z)

int test_vtranspose()
{
#ifdef FLOAT
    float __attribute__((aligned(32))) m[64];
    for (int i = 0; i < 64; ++i) {
        m[i] = ((float) rand()) / RAND_MAX;
    }

    __m256 r0 = _mm256_load_ps(m);
    __m256 r1 = _mm256_load_ps(m + 8);
    __m256 r2 = _mm256_load_ps(m + 16);
    __m256 r3 = _mm256_load_ps(m + 24);
    __m256 r4 = _mm256_load_ps(m + 32);
    __m256 r5 = _mm256_load_ps(m + 40);
    __m256 r6 = _mm256_load_ps(m + 48);
    __m256 r7 = _mm256_load_ps(m + 56);

    vtranspose(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);

    float __attribute__((aligned(32))) t[64];

    _mm256_store_ps(t, r0);
    _mm256_store_ps(t + 8, r1);
    _mm256_store_ps(t + 16, r2);
    _mm256_store_ps(t + 24, r3);
    _mm256_store_ps(t + 32, r4);
    _mm256_store_ps(t + 40, r5);
    _mm256_store_ps(t + 48, r6);
    _mm256_store_ps(t + 56, r7);

    int status = SUCCESS;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            if (m[i * 8 + j] != t[j * 8 + i]) {
                status = FAILURE;
            }
        }
    }

    return status;
#else
    double __attribute__((aligned(32))) m[16];
    for (int i = 0; i < 16; ++i) {
        m[i] = ((double) rand()) / RAND_MAX;
    }

    __m256d r0 = _mm256_load_pd(m);
    __m256d r1 = _mm256_load_pd(m + 4);
    __m256d r2 = _mm256_load_pd(m + 8);
    __m256d r3 = _mm256_load_pd(m + 12);

    vtranspose(&r0, &r1, &r2, &r3);

    double __attribute__((aligned(32))) t[16];

    _mm256_store_pd(t, r0);
    _mm256_store_pd(t + 4, r1);
    _mm256_store_pd(t + 8, r2);
    _mm256_store_pd(t + 12, r3);

    int status = SUCCESS;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (m[i * 4 + j] != t[j * 4 + i]) {
                status = FAILURE;
            }
        }
    }

    return status;
#endif
}

#define RMJ_IDX(z, y, x, h, w) ((h) * (w) * (z) + (w) * (y) + (x))

static int verify_wDxx_rhs_comp(const ftype *__restrict__ k,
                                const ftype *__restrict__ D_pp,
                                const ftype *__restrict__ eta,
                                const ftype *__restrict__ zeta,
                                const ftype *__restrict__ u,
                                const ftype *__restrict__ rhs,
                                int depth,
                                int height,
                                int width,
                                ftype nu,
                                ftype dt,
                                ftype dx,
                                ftype tol)
{
    uint64_t size = depth * height * width;
    ftype *Dxx_eta = malloc(size * sizeof(ftype));
    ftype *Dyy_zeta = malloc(size * sizeof(ftype));
    ftype *Dzz_u = malloc(size * sizeof(ftype));

    /* WARNING: We need signed ints here, otherwise indexes underflow. */

    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < width; ++k) {
                uint64_t idx = RMJ_IDX(i, j, k, height, width);
                /* Compute second derivatives. */
                Dxx_eta[idx] = (eta[RMJ_IDX(i, j, k - 1, height, width)] -
                                2 * eta[idx] +
                                eta[RMJ_IDX(i, j, k + 1, height, width)]) /
                               (dx * dx);
                Dyy_zeta[idx] = (zeta[RMJ_IDX(i, j - 1, k, height, width)] -
                                 2 * zeta[idx] +
                                 zeta[RMJ_IDX(i, j + 1, k, height, width)]) /
                                (dx * dx);
                Dzz_u[idx] = (u[RMJ_IDX(i - 1, j, k, height, width)] -
                              2 * u[idx] +
                              u[RMJ_IDX(i + 1, j, k, height, width)]) /
                             (dx * dx);
            }
        }
    }

    int status = SUCCESS;
    for (uint64_t i = 0; i < size && status == SUCCESS; ++i) {
        /* WARNING: Null volume force. */
        ftype f = 0;
        ftype g = f - D_pp[i] - nu / (2 * k[i]) * u[i] +
                  (nu / 2) * (Dxx_eta[i] + Dyy_zeta[i] + Dzz_u[i]);
        ftype beta = 1 + dt * nu / (2 * k[i]);
        ftype rhs_ref = u[i] + dt / beta * g - eta[i];

        if (abs(rhs_ref - rhs[i]) > tol) {
            status = FAILURE;
        }
    }

    free(Dzz_u);
    free(Dyy_zeta);
    free(Dxx_eta);

    return status;
}

int test_wDxx_rhs_computation(uint32_t depth,
                              uint32_t height,
                              uint32_t width)
{
    ASSERT_TRUE(width % VLEN == 0);

    size_t size = (depth + 2) * height * width;
    ftype *k = aligned_alloc(32, size * sizeof(ftype));
    ftype *p = aligned_alloc(32, size * sizeof(ftype));
    ftype *phi = aligned_alloc(32, size * sizeof(ftype));
    ftype *eta = aligned_alloc(32, 3 * size * sizeof(ftype));
    ftype *zeta = aligned_alloc(32, 3 * size * sizeof(ftype));
    ftype *u = aligned_alloc(32, 3 * size * sizeof(ftype));
    ftype *rhs = aligned_alloc(32, 3 * size * sizeof(ftype));

    rand_fill(k, size);
    rand_fill(p, size);
    rand_fill(phi, size);
    rand_fill(eta, 3 * size);
    rand_fill(zeta, 3 * size);
    rand_fill(u, 3 * size);

    uint32_t face_size = height * width;

    ftype *eta_x = eta + face_size;
    ftype *eta_y = eta + face_size + size;
    ftype *eta_z = eta + face_size + 2 * size;
    ftype *zeta_x = zeta + face_size;
    ftype *zeta_y = zeta + face_size + size;
    ftype *zeta_z = zeta + face_size + 2 * size;
    ftype *u_x = u + face_size;
    ftype *u_y = u + face_size + size;
    ftype *u_z = u + face_size + 2 * size;
    ftype *rhs_x = rhs;
    ftype *rhs_y = rhs + size;
    ftype *rhs_z = rhs + 2 * size;

    ftype nu = ((ftype) rand()) / RAND_MAX;
    ftype dt = ((ftype) rand()) / RAND_MAX;
    ftype dx = ((ftype) rand()) / RAND_MAX;

    compute_wDxx_rhs(k, p, phi,
                     eta_x, eta_y, eta_z,
                     zeta_x, zeta_y, zeta_z,
                     u_x, u_y, u_z,
                     depth, height, width,
                     nu,
                     dt,
                     dx,
                     rhs_x, rhs_y, rhs_z);

    uint64_t num_points = depth * height * width;
    ftype *pp = malloc((num_points + height * width) * sizeof(ftype));
    ftype *Dx_pp = malloc(num_points * sizeof(ftype));
    ftype *Dy_pp = malloc(num_points * sizeof(ftype));
    ftype *Dz_pp = malloc(num_points * sizeof(ftype));
    memset(Dx_pp, 0, num_points * sizeof(ftype));
    memset(Dy_pp, 0, num_points * sizeof(ftype));
    memset(Dz_pp, 0, num_points * sizeof(ftype));

    /* Compute pressure predictor. */
    for (uint64_t i = 0; i < num_points + height * width; ++i) {
        pp[i] = p[i] + phi[i];
    }
    /* Compute pressure predictor gradient. */
    for (uint32_t i = 0; i < depth; ++i) {
        for (uint32_t j = 0; j < height; ++j) {
            for (uint32_t k = 0; k < width; ++k) {
                uint64_t idx = RMJ_IDX(i, j, k, height, width);
                ftype pp_ijk = pp[idx];
                ftype Dx_pp_ijk = pp[RMJ_IDX(i, j, k + 1, height, width)] -
                                  pp_ijk;
                ftype Dy_pp_ijk = pp[RMJ_IDX(i, j + 1, k, height, width)] -
                                  pp_ijk;
                ftype Dz_pp_ijk = pp[RMJ_IDX(i + 1, j, k, height, width)] -
                                  pp_ijk;
                Dx_pp[idx] = Dx_pp_ijk / dx;
                Dy_pp[idx] = (j == height - 1) ? 0 : Dy_pp_ijk / dx;
                Dz_pp[idx] = (i == depth - 1) ? 0 : Dz_pp_ijk / dx;
            }
        }
    }

    int error = verify_wDxx_rhs_comp(k, Dx_pp, eta_x, zeta_x,
                                     u_x, rhs_x, depth, height,
                                     width, nu, dt, dx, 1e-4) ||
                verify_wDxx_rhs_comp(k, Dy_pp, eta_y, zeta_y,
                                     u_y, rhs_y, depth, height,
                                     width, nu, dt, dx, 1e-4) ||
                verify_wDxx_rhs_comp(k, Dz_pp, eta_z, zeta_z,
                                     u_z, rhs_z, depth, height,
                                     width, nu, dt, dx, 1e-4);

    free(Dz_pp);
    free(Dy_pp);
    free(Dx_pp);
    free(pp);
    free(rhs);
    free(u);
    free(zeta);
    free(eta);
    free(phi);
    free(p);
    free(k);

    return error;
}

int main(void)
{
    srand(SEED);

    EXPECT_SUCCESS(test_vtranspose());

    EXPECT_SUCCESS(test_wDxx_rhs_computation(1, 32, 64));
    EXPECT_SUCCESS(test_wDxx_rhs_computation(32, 64, 128));
    EXPECT_SUCCESS(test_wDxx_rhs_computation(128, 128, 64));

    EXPECT_SUCCESS(test_momentum_Dxx_solver(1, 16, 16));
    EXPECT_SUCCESS(test_momentum_Dxx_solver(128, 128, 64));
    EXPECT_SUCCESS(test_momentum_Dxx_solver(512, 32, 256));

    EXPECT_SUCCESS(test_momentum_Dyy_solver(32, 64, 64));
    EXPECT_SUCCESS(test_momentum_Dyy_solver(128, 128, 64));
    EXPECT_SUCCESS(test_momentum_Dyy_solver(512, 32, 256));

    EXPECT_SUCCESS(test_momentum_Dzz_solver(32, 64, 64));
    EXPECT_SUCCESS(test_momentum_Dzz_solver(128, 128, 64));
    EXPECT_SUCCESS(test_momentum_Dzz_solver(32, 32, 256));

    EXPECT_SUCCESS(test_pressure_Dxx_solver(1, 32, 32));
    EXPECT_SUCCESS(test_pressure_Dxx_solver(32, 64, 128));
    EXPECT_SUCCESS(test_pressure_Dxx_solver(128, 64, 512));

    EXPECT_SUCCESS(test_pressure_Dyy_solver(1, 32, 32));
    EXPECT_SUCCESS(test_pressure_Dyy_solver(32, 64, 128));
    EXPECT_SUCCESS(test_pressure_Dyy_solver(128, 64, 512));

    EXPECT_SUCCESS(test_pressure_Dzz_solver(64, 32, 32));
    EXPECT_SUCCESS(test_pressure_Dzz_solver(256, 32, 128));
    EXPECT_SUCCESS(test_pressure_Dzz_solver(128, 64, 64));

    return 0;
}
