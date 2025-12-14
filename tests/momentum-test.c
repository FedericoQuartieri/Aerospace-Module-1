#include "ftype.h"
#include "alloc.h"
#include "field.h"
#include "consts.h"

#include "test.h"
#include "lin-solver-test.h"

#include "momentum.c"

#define TOL 1e-4

#define RMJ_IDX(z, y, x, h, w) ((h) * (w) * (z) + (w) * (y) + (x))

static DEF_TEST(test_momentum_Dxx_rhs_comp,
                ArenaAllocator *arena,
                const ftype *restrict k,
                const ftype *restrict D_pp,
                const ftype *restrict eta,
                const ftype *restrict zeta,
                const ftype *restrict u,
                const ftype *restrict rhs,
                int depth,
                int height,
                int width,
                ftype u_ex,
                int is_x_comp,
                int is_y_comp,
                int is_z_comp)
{
    arena_enter(arena);

    uint64_t size = depth * height * width;
    ftype *Dxx_eta = arena_push_count(arena, ftype, size);
    ftype *Dyy_zeta = arena_push_count(arena, ftype, size);
    ftype *Dzz_u = arena_push_count(arena, ftype, size);

    /* WARNING: We need signed ints here, otherwise indexes underflow. */
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int l = 0; l < width; ++l) {
                uint64_t idx = RMJ_IDX(i, j, l, height, width);
                /* Compute second derivatives. */
                Dxx_eta[idx] = (eta[RMJ_IDX(i, j, l - 1, height, width)] -
                                2 * eta[idx] +
                                eta[RMJ_IDX(i, j, l + 1, height, width)]) /
                               (_DX * _DX);
                Dyy_zeta[idx] = (zeta[RMJ_IDX(i, j - 1, l, height, width)] -
                                 2 * zeta[idx] +
                                 zeta[RMJ_IDX(i, j + 1, l, height, width)]) /
                                (_DX * _DX);
                Dzz_u[idx] = (u[RMJ_IDX(i - 1, j, l, height, width)] -
                              2 * u[idx] +
                              u[RMJ_IDX(i + 1, j, l, height, width)]) /
                             (_DX * _DX);

                /* Ghost node interpolation. */
                if (!is_x_comp && l == width - 1) {
                    Dxx_eta[idx] = (eta[RMJ_IDX(i, j, l - 1, height, width)] -
                                    3 * eta[idx] + 2 * u_ex) / (_DX * _DX);
                }

                if (!is_y_comp && j == height - 1) {
                    Dyy_zeta[idx] = (zeta[RMJ_IDX(i, j - 1, l, height, width)] -
                                     3 * zeta[idx] + 2 * u_ex) / (_DX * _DX);
                }

                if (!is_z_comp && i == depth - 1) {
                    Dzz_u[idx] = (u[RMJ_IDX(i - 1, j, l, height, width)] -
                                  3 * u[idx] + 2 * u_ex) / (_DX * _DX);
                }
            }
        }
    }

    for (uint32_t i = 0; i < depth; ++i) {
        for (uint32_t j = 0; j < height; ++j) {
            for (uint32_t l = 0; l < width; ++l) {

                if (i == 0 || j == 0 || l == 0 ||
                    (is_x_comp && l == width - 1) ||
                    (is_y_comp && j == height - 1) ||
                    (is_z_comp && i == depth - 1))
                {
                    /* No need to compute rhs here. */

                } else {
                    uint64_t idx = height * width * i + width * j + l;

                    /* WARNING: Null volume force. */
                    ftype f = 0;
                    ftype g = f - D_pp[idx] - _NU / (2 * k[idx]) * u[idx] +
                              (_NU / 2) * (Dxx_eta[idx] + Dyy_zeta[idx] +
                                           Dzz_u[idx]);
                    ftype beta = 1 + _DT * _NU / (2 * k[idx]);
                    ftype rhs_ref = u[idx] + _DT / beta * g - eta[idx];

                    EXPECT_EQUALF(rhs_ref, rhs[idx], TOL);
                }
           }
        }
    }

    arena_exit(arena);
}

DEF_TEST(test_momentum_Dxx_rhs,
         ArenaAllocator *arena,
         int depth,
         int height,
         int width)
{
    arena_enter(arena);

    field_size size = { width, height, (depth + 2) };
    field porosity = field_alloc(size, arena);
    field pressure = field_alloc(size, arena);
    field pressure_delta = field_alloc(size, arena);
    field3 velocity_Dxx = field3_alloc(size, arena);
    field3 velocity_Dyy = field3_alloc(size, arena);
    field3 velocity_Dzz = field3_alloc(size, arena);
    field3 forcing = field3_alloc(size, arena);
    field3 rhs = field3_alloc(size, arena);

    field_rand_fill(size, porosity);
    field_rand_fill(size, pressure);
    field_rand_fill(size, pressure_delta);
    field3_rand_fill(size, velocity_Dxx);
    field3_rand_fill(size, velocity_Dyy);
    field3_rand_fill(size, velocity_Dzz);
    field3_fill(size, 0, forcing);

    ftype u_ex_x = ((ftype) rand()) / RAND_MAX;
    ftype u_ex_y = ((ftype) rand()) / RAND_MAX;
    ftype u_ex_z = ((ftype) rand()) / RAND_MAX;

    uint64_t face_size = height * width;
    compute_Dxx_rhs(porosity,
                    pressure, pressure_delta,
                    velocity_Dxx.x + face_size,
                    velocity_Dxx.y + face_size,
                    velocity_Dxx.z + face_size,
                    velocity_Dyy.x + face_size,
                    velocity_Dyy.y + face_size,
                    velocity_Dyy.z + face_size,
                    velocity_Dzz.x + face_size,
                    velocity_Dzz.y + face_size,
                    velocity_Dzz.z + face_size,
                    forcing.x, forcing.y, forcing.z,
                    depth, height, width,
                    u_ex_x, u_ex_y, u_ex_z,
                    rhs.x, rhs.y, rhs.z);

    /* Pressure predictor. */
    field pressure_pred = field_alloc(size, arena);
    field Dx_pressure_pred = field_alloc(size, arena);
    field Dy_pressure_pred = field_alloc(size, arena);
    field Dz_pressure_pred = field_alloc(size, arena);
    field_fill(size, 0, Dx_pressure_pred);
    field_fill(size, 0, Dy_pressure_pred);
    field_fill(size, 0, Dz_pressure_pred);

    uint64_t num_points = depth * height * width;
    for (uint64_t i = 0; i < num_points + face_size; ++i) {
        pressure_pred[i] = pressure[i] + pressure_delta[i];
    }
    /* Compute pressure predictor gradient. */
    for (uint32_t i = 0; i < depth; ++i) {
        for (uint32_t j = 0; j < height; ++j) {
            for (uint32_t l = 0; l < width; ++l) {
                uint64_t idx = RMJ_IDX(i, j, l, height, width);
                ftype pressure_pred_ijk = pressure_pred[idx];
                ftype Dx_pressure_pred_ijk =
                    pressure_pred[RMJ_IDX(i, j, l + 1, height, width)] -
                    pressure_pred_ijk;
                ftype Dy_pressure_pred_ijk =
                    pressure_pred[RMJ_IDX(i, j + 1, l, height, width)] -
                    pressure_pred_ijk;
                ftype Dz_pressure_pred_ijk =
                    pressure_pred[RMJ_IDX(i + 1, j, l, height, width)] -
                    pressure_pred_ijk;
                /* Pressure gradient is zero at the right boundaries. */
                Dx_pressure_pred[idx] =
                    (l == width - 1) ? 0 : Dx_pressure_pred_ijk / _DX;
                Dy_pressure_pred[idx] =
                    (j == height - 1) ? 0 : Dy_pressure_pred_ijk / _DX;
                Dz_pressure_pred[idx] =
                    (i == depth - 1) ? 0 : Dz_pressure_pred_ijk / _DX;
            }
        }
    }

    EXPECT_SUCCESS(test_momentum_Dxx_rhs_comp,
                   arena,
                   porosity,
                   Dx_pressure_pred,
                   velocity_Dxx.x + face_size,
                   velocity_Dyy.x + face_size,
                   velocity_Dzz.x + face_size,
                   rhs.x,
                   depth, height, width,
                   u_ex_x, 1, 0, 0);

    EXPECT_SUCCESS(test_momentum_Dxx_rhs_comp,
                   arena,
                   porosity,
                   Dy_pressure_pred,
                   velocity_Dxx.y + face_size,
                   velocity_Dyy.y + face_size,
                   velocity_Dzz.y + face_size,
                   rhs.y,
                   depth, height, width,
                   u_ex_y, 0, 1, 0);

    EXPECT_SUCCESS(test_momentum_Dxx_rhs_comp,
                   arena,
                   porosity,
                   Dz_pressure_pred,
                   velocity_Dxx.z + face_size,
                   velocity_Dyy.z + face_size,
                   velocity_Dzz.z + face_size,
                   rhs.z,
                   depth, height, width,
                   u_ex_z, 0, 0, 1);

    arena_exit(arena);
}

#define DEF_TEST_MOMENTUM_D_SOLVER(                                      \
    axes, is_x_normal, is_y_normal, is_z_normal,                         \
    u0_x, u0_y, u0_z, un_x, un_y, un_z)                                  \
DEF_TEST(test_momentum_D##axes##_solver,                                 \
         ArenaAllocator *arena,                                          \
         uint32_t depth,                                                 \
         uint32_t height,                                                \
         uint32_t width)                                                 \
{                                                                        \
    arena_enter(arena);                                                  \
                                                                         \
    field_size size = { width, height, depth };                          \
    field gamma = field_alloc(size, arena);                              \
    /* WARNING: size may be smaller than 4 * width * VLEN */             \
    field tmp = field_alloc(size, arena);                                \
    field3 sol = field3_alloc(size, arena);                              \
    field3 rhs = field3_alloc(size, arena);                              \
    field3 rhs_ref = field3_alloc(size, arena);                          \
                                                                         \
    field_rand_fill(size, gamma);                                        \
    field3_rand_fill(size, rhs);                                         \
    field3_copy(size, to_const_field3(rhs), rhs_ref);                    \
                                                                         \
    solve_D##axes##_blocks(gamma, depth, height, width,                  \
                           tmp, rhs.x, rhs.y, rhs.z,                     \
                           sol.x, sol.y, sol.z);                         \
                                                                         \
    EXPECT_SUCCESS(test_wD##axes##_solution,                             \
                   gamma, rhs_ref.x, sol.x, u0_x, un_x,                  \
                   depth, height, width, is_x_normal, 1);                \
                                                                         \
    EXPECT_SUCCESS(test_wD##axes##_solution,                             \
                   gamma, rhs_ref.y, sol.y, u0_y, un_y,                  \
                   depth, height, width, is_y_normal, 1);                \
                                                                         \
    EXPECT_SUCCESS(test_wD##axes##_solution,                             \
                   gamma, rhs_ref.z, sol.z, u0_z, un_z,                  \
                   depth, height, width, is_z_normal, 1);                \
                                                                         \
    arena_exit(arena);                                                   \
}

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

DEF_TEST_MOMENTUM_D_SOLVER(
    xx, 1, 0, 0, LEFT_BC_U_X, LEFT_BC_U_Y, LEFT_BC_U_Z,
    RIGHT_BC_U_X, RIGHT_BC_U_Y, RIGHT_BC_U_Z)

DEF_TEST_MOMENTUM_D_SOLVER(
    yy, 0, 1, 0, TOP_BC_U_X, TOP_BC_U_Y, TOP_BC_U_Z,
    BOTTOM_BC_U_X, BOTTOM_BC_U_Y, BOTTOM_BC_U_Z)

DEF_TEST_MOMENTUM_D_SOLVER(
    zz, 0, 0, 1, FRONT_BC_U_X, FRONT_BC_U_Y, FRONT_BC_U_Z,
    BACK_BC_U_X, BACK_BC_U_Y, BACK_BC_U_Z)
