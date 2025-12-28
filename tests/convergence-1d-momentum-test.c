#include <math.h>

#include "test.h"
#include "ftype.h"
#include "alloc.h"
#include "field.h"
#include "consts.h"
#include "convergence-test.h"

#include "momentum.c"

DEFINE_DX(1.0)
DEFINE_DT(1.0)

static ftype get_man_u_x(ftype x, ftype y, ftype z, ftype t)
{
    return sin(t) * sin(x) * sin(y) * sin(z);
}

static ftype get_man_u_y(ftype x, ftype y, ftype z, ftype t)
{
    return sin(t) * cos(x) * cos(y) * cos(z);
}

static ftype get_man_u_z(ftype x, ftype y, ftype z, ftype t)
{
    return sin(t) * cos(x) * sin(y) * (cos(z) + sin(z));
}

static inline void left_bc_manufactured(uint32_t x,
                                        uint32_t y,
                                        uint32_t z,
                                        uint32_t t,
                                        vftype *restrict u_x,
                                        vftype *restrict u_y,
                                        vftype *restrict u_z)
{
    /* On the left boundary, we are vectorizing across rows. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        /* y and z are on the wall */
        tmp_y[i] = get_man_u_y(0, (y + i) * _DX + _DX / 2, z * _DX, t * _DT);
        tmp_z[i] = get_man_u_z(0, (y + i) * _DX, z * _DX + _DX / 2, t * _DT);

        /* u_1/2 = u0 + dux/dx DX/2 = u0 - (duy/dy + duz/dz) DX/2 */

        ftype duy_dy =
            get_man_u_y(0, (y + i) * _DX + _DX / 2, z * _DX, t * _DT) -
            get_man_u_y(0, (y + i) * _DX - _DX / 2, z * _DX, t * _DT);

        ftype duz_dz =
            get_man_u_z(0, (y + i) * _DX, z * _DX + _DX / 2, t * _DT) -
            get_man_u_z(0, (y + i) * _DX, z * _DX - _DX / 2, t * _DT);

        tmp_x[i] = get_man_u_x(0, (y + i) * _DX, z * _DX, t * _DT) -
                   (duy_dy + duz_dz) / 2; /* Already multiplied by _DX */
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline void top_bc_manufactured(uint32_t x,
                                       uint32_t y,
                                       uint32_t z,
                                       uint32_t t,
                                       vftype *restrict u_x,
                                       vftype *restrict u_y,
                                       vftype *restrict u_z)
{
    /* On the top boundary, we are vectorizing across columns. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        tmp_x[i] = get_man_u_x((x + i) * _DX + _DX / 2, 0, z * _DX, t * _DT);
        tmp_z[i] = get_man_u_z((x + i) * _DX, 0, z * _DX + _DX / 2, t * _DT);

        ftype dux_dx =
            (get_man_u_x((x + i) * _DX + _DX / 2, 0, z * _DX, t * _DT) -
             get_man_u_x((x + i) * _DX - _DX / 2, 0, z * _DX, t * _DT));

        ftype duz_dz =
            (get_man_u_z((x + i) * _DX, 0, z * _DX + _DX / 2, t * _DT) -
             get_man_u_z((x + i) * _DX, 0, z * _DX - _DX / 2, t * _DT));

        tmp_y[i] = get_man_u_y((x + i) * _DX, 0, z * _DX, t * _DT) -
                   (dux_dx + duz_dz) / 2;
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline void front_bc_manufactured(uint32_t x,
                                         uint32_t y,
                                         uint32_t z,
                                         uint32_t t,
                                         vftype *restrict u_x,
                                         vftype *restrict u_y,
                                         vftype *restrict u_z)
{
    /* On the front boundary, we are vectorizing across columns. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        tmp_x[i] = get_man_u_x((x + i) * _DX + _DX / 2, y * _DX, 0, t * _DT);
        tmp_y[i] = get_man_u_y((x + i) * _DX, y * _DX + _DX / 2, 0, t * _DT);

        ftype dux_dx =
            (get_man_u_x((x + i) * _DX + _DX / 2, y * _DX, 0, t * _DT) -
             get_man_u_x((x + i) * _DX - _DX / 2, y * _DX, 0, t * _DT));

        ftype duy_dy =
            (get_man_u_y((x + i) * _DX, y * _DX + _DX / 2, 0, t * _DT) -
             get_man_u_y((x + i) * _DX, y * _DX - _DX / 2, 0, t * _DT));

        tmp_z[i] = get_man_u_z((x + i) * _DX, y * _DX, 0, t * _DT) -
                   (dux_dx + duy_dy) / 2; /* Already multiplied by _DX */
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline void right_bc_manufactured(uint32_t x,
                                         uint32_t y,
                                         uint32_t z,
                                         uint32_t t,
                                         vftype *restrict u_x,
                                         vftype *restrict u_y,
                                         vftype *restrict u_z)
{
    /* On the front boundary, we are vectorizing across columns. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        tmp_x[i] = get_man_u_x(x * _DX + _DX / 2, (y + i) * _DX, z * _DX, t * _DT);
        tmp_y[i] = get_man_u_y(x * _DX + _DX / 2, (y + i) * _DX + _DX / 2, z * _DX, t * _DT);
        tmp_z[i] = get_man_u_z(x * _DX + _DX / 2, (y + i) * _DX, z * _DX + _DX / 2, t * _DT);
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline void bottom_bc_manufactured(uint32_t x,
                                          uint32_t y,
                                          uint32_t z,
                                          uint32_t t,
                                          vftype *restrict u_x,
                                          vftype *restrict u_y,
                                          vftype *restrict u_z)
{
    /* On the front boundary, we are vectorizing across columns. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        tmp_x[i] = get_man_u_x((x + i) * _DX + _DX / 2, y * _DX + _DX / 2, z * _DX, t * _DT);
        tmp_y[i] = get_man_u_y((x + i) * _DX, y * _DX + _DX / 2, z * _DX, t * _DT);
        tmp_z[i] = get_man_u_z((x + i) * _DX, y * _DX + _DX / 2, z * _DX + _DX / 2, t * _DT);
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline void back_bc_manufactured(uint32_t x,
                                        uint32_t y,
                                        uint32_t z,
                                        uint32_t t,
                                        vftype *restrict u_x,
                                        vftype *restrict u_y,
                                        vftype *restrict u_z)
{
    /* On the front boundary, we are vectorizing across columns. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        tmp_x[i] = get_man_u_x((x + i) * _DX + _DX / 2, y * _DX, z * _DX + _DX / 2, t * _DT);
        tmp_y[i] = get_man_u_y((x + i) * _DX, y * _DX + _DX / 2, z * _DX + _DX / 2, t * _DT);
        tmp_z[i] = get_man_u_z((x + i) * _DX, y * _DX, z * _DX + _DX / 2, t * _DT);
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline void left_bc_manufactured_delta(uint32_t x,
                                              uint32_t y,
                                              uint32_t z,
                                              uint32_t t,
                                              vftype *restrict u_x,
                                              vftype *restrict u_y,
                                              vftype *restrict u_z)
{
    vftype u_x_prev, u_y_prev, u_z_prev;
    left_bc_manufactured(x, y, z, t - 1, &u_x_prev, &u_y_prev, &u_z_prev);

    left_bc_manufactured(x, y, z, t, u_x, u_y, u_z);

    *u_x = *u_x - u_x_prev;
    *u_y = *u_y - u_y_prev;
    *u_z = *u_z - u_z_prev;
}

static inline void right_bc_manufactured_delta(uint32_t x,
                                               uint32_t y,
                                               uint32_t z,
                                               uint32_t t,
                                               vftype *restrict u_x,
                                               vftype *restrict u_y,
                                               vftype *restrict u_z)
{
    vftype u_x_prev, u_y_prev, u_z_prev;
    right_bc_manufactured(x, y, z, t - 1, &u_x_prev, &u_y_prev, &u_z_prev);

    right_bc_manufactured(x, y, z, t, u_x, u_y, u_z);

    *u_x = *u_x - u_x_prev;
    *u_y = *u_y - u_y_prev;
    *u_z = *u_z - u_z_prev;
}

static inline void top_bc_manufactured_delta(uint32_t x,
                                             uint32_t y,
                                             uint32_t z,
                                             uint32_t t,
                                             vftype *restrict u_x,
                                             vftype *restrict u_y,
                                             vftype *restrict u_z)
{
    vftype u_x_prev, u_y_prev, u_z_prev;
    top_bc_manufactured(x, y, z, t - 1, &u_x_prev, &u_y_prev, &u_z_prev);

    top_bc_manufactured(x, y, z, t, u_x, u_y, u_z);

    *u_x = *u_x - u_x_prev;
    *u_y = *u_y - u_y_prev;
    *u_z = *u_z - u_z_prev;
}

static inline void bottom_bc_manufactured_delta(uint32_t x,
                                                uint32_t y,
                                                uint32_t z,
                                                uint32_t t,
                                                vftype *restrict u_x,
                                                vftype *restrict u_y,
                                                vftype *restrict u_z)
{
    vftype u_x_prev, u_y_prev, u_z_prev;
    bottom_bc_manufactured(x, y, z, t - 1, &u_x_prev, &u_y_prev, &u_z_prev);

    bottom_bc_manufactured(x, y, z, t, u_x, u_y, u_z);

    *u_x = *u_x - u_x_prev;
    *u_y = *u_y - u_y_prev;
    *u_z = *u_z - u_z_prev;
}

static inline void front_bc_manufactured_delta(uint32_t x,
                                               uint32_t y,
                                               uint32_t z,
                                               uint32_t t,
                                               vftype *restrict u_x,
                                               vftype *restrict u_y,
                                               vftype *restrict u_z)
{
    vftype u_x_prev, u_y_prev, u_z_prev;
    front_bc_manufactured(x, y, z, t - 1, &u_x_prev, &u_y_prev, &u_z_prev);

    front_bc_manufactured(x, y, z, t, u_x, u_y, u_z);

    *u_x = *u_x - u_x_prev;
    *u_y = *u_y - u_y_prev;
    *u_z = *u_z - u_z_prev;
}

static inline void back_bc_manufactured_delta(uint32_t x,
                                              uint32_t y,
                                              uint32_t z,
                                              uint32_t t,
                                              vftype *restrict u_x,
                                              vftype *restrict u_y,
                                              vftype *restrict u_z)
{
    vftype u_x_prev, u_y_prev, u_z_prev;
    back_bc_manufactured(x, y, z, t - 1, &u_x_prev, &u_y_prev, &u_z_prev);

    back_bc_manufactured(x, y, z, t, u_x, u_y, u_z);

    *u_x = *u_x - u_x_prev;
    *u_y = *u_y - u_y_prev;
    *u_z = *u_z - u_z_prev;
}

DEFINE_FUNCTION_BC_U(left_bc_manufactured, left_bc_manufactured_delta, BC_LEFT)
DEFINE_FUNCTION_BC_U(top_bc_manufactured, top_bc_manufactured_delta, BC_TOP)
DEFINE_FUNCTION_BC_U(front_bc_manufactured, front_bc_manufactured_delta, BC_FRONT)
DEFINE_FUNCTION_BC_U(right_bc_manufactured, right_bc_manufactured_delta, BC_RIGHT)
DEFINE_FUNCTION_BC_U(bottom_bc_manufactured, bottom_bc_manufactured_delta, BC_BOTTOM)
DEFINE_FUNCTION_BC_U(back_bc_manufactured, back_bc_manufactured_delta, BC_BACK)

static void compute_manufactured_forcing(field_size size, field3 dst)
{
    for (uint32_t i = 0; i < size.depth; ++i) {
        for (uint32_t j = 0; j < size.height; ++j) {
            for (uint32_t k = 0; k < size.width; ++k) {
                uint64_t idx = size.height * size.width * i +
                               size.width * j + k;

                dst.x[idx] = 2 * get_man_u_x(_DX * k + _DX / 2, _DX * j, _DX * i, _DT);
                dst.y[idx] = 2 * get_man_u_y(_DX * k, _DX * j + _DX / 2, _DX * i, _DT);
                dst.z[idx] = 2 * get_man_u_z(_DX * k, _DX * j, _DX * i + _DX / 2, _DT);
            }
        }
    }
}

static void compute_manufactured_solution(field_size size,
                                          uint32_t timestep,
                                          field3 dst)
{
    ftype time = timestep * _DT;
    for (uint32_t i = 0; i < size.depth; ++i) {
        for (uint32_t j = 0; j < size.height; ++j) {
            for (uint32_t k = 0; k < size.width; ++k) {
                uint64_t idx = size.height * size.width * i +
                               size.width * j + k;

                dst.x[idx] = get_man_u_x(_DX * k + _DX / 2, _DX * j, _DX * i, time);
                dst.y[idx] = get_man_u_y(_DX * k, _DX * j + _DX / 2, _DX * i, time);
                dst.z[idx] = get_man_u_z(_DX * k, _DX * j, _DX * i + _DX / 2, time);
            }
        }
    }
}

#define DEF_TEST_CONVERGENCE_SPACE_MOMENTUM_SOLVER_D(axis)                   \
DEF_TEST(test_convergence_space_momentum_solver_D##axis,                     \
         ArenaAllocator *arena,                                              \
         int min_depth,                                                      \
         int min_height,                                                     \
         int min_width,                                                      \
         int num_samples)                                                    \
{                                                                            \
    arena_enter(arena);                                                      \
                                                                             \
    double *errors = arena_push_count(arena, double, num_samples);           \
    double *dxs = arena_push_count(arena, double, num_samples);              \
                                                                             \
    for (int i = 0; i < num_samples; ++i) {                                  \
        arena_enter(arena);                                                  \
                                                                             \
        field_size size = { min_width << i,                                  \
                            min_height << i,                                 \
                            min_depth << i };                                \
                                                                             \
        SET_DX(1.0 / size.width);                                            \
                                                                             \
        field tmp = field_alloc(size, arena);                                \
        field gamma = field_alloc(size, arena);                              \
        field_fill(size, 1.0 / (_DX * _DX), gamma);                          \
                                                                             \
        field3 solution = field3_alloc(size, arena);                         \
        field3 forcing = field3_alloc(size, arena);                          \
        field3 manufactured = field3_alloc(size, arena);                     \
        compute_manufactured_solution(size, 1, manufactured);                \
        compute_manufactured_forcing(size, forcing);                         \
                                                                             \
        solve_D##axis##_blocks(gamma, size.depth, size.height, size.width,   \
                               1, tmp, forcing.x, forcing.y, forcing.z,      \
                               solution.x, solution.y, solution.z);          \
                                                                             \
        dxs[i] = _DX;                                                        \
        errors[i] = field3_l2_norm_diff(size, _DX, to_const_field3(solution),\
                                        to_const_field3(manufactured));      \
                                                                             \
        arena_exit(arena);                                                   \
    }                                                                        \
                                                                             \
    double *orders = arena_push_count(arena, double, num_samples);           \
    estimate_convergence_order(errors, dxs, num_samples, orders);            \
                                                                             \
    printf("solve_D" #axis "_blocks()\n");                                   \
    print_convergence_table(errors, dxs, orders, num_samples);               \
                                                                             \
    for (int i = 1; i < num_samples; ++i) {                                  \
        EXPECT_EQUALF(orders[i], 2.0, 0.05);                                 \
    }                                                                        \
                                                                             \
    arena_exit(arena);                                                       \
}

DEF_TEST_CONVERGENCE_SPACE_MOMENTUM_SOLVER_D(xx)
DEF_TEST_CONVERGENCE_SPACE_MOMENTUM_SOLVER_D(yy)
DEF_TEST_CONVERGENCE_SPACE_MOMENTUM_SOLVER_D(zz)

static void compute_manufactured_forcing_unsteady(field_size size,
                                                  uint32_t timestep,
                                                  field3 dst)
{
    ftype t1 = (timestep - 1) * _DT;
    ftype t2 = timestep * _DT;

    for (uint32_t i = 0; i < size.depth; ++i) {
        for (uint32_t j = 0; j < size.height; ++j) {
            for (uint32_t k = 0; k < size.width; ++k) {
                uint64_t idx = size.height * size.width * i +
                               size.width * j + k;

                
                dst.x[idx] = 0.5 * (cos(t1) + sin(t1) + cos(t2) + sin(t2))
                                 * sin(k * _DX + _DX / 2)
                                 * sin(j * _DX)
                                 * sin(i * _DX);

                dst.y[idx] = 0.5 * (cos(t1) + sin(t1) + cos(t2) + sin(t2))
                                 * cos(k * _DX)
                                 * cos(j * _DX + _DX / 2)
                                 * cos(i * _DX);

                dst.z[idx] = 0.5 * (cos(t1) + sin(t1) + cos(t2) + sin(t2))
                                 * cos(k * _DX)
                                 * sin(j * _DX)
                                 * (cos(i * _DX + _DX / 2) +
                                   sin(i * _DX + _DX / 2));
            }
        }
    }
}

DEF_TEST(test_convergence_time_momentum_solver_Dxx,
         ArenaAllocator *arena,
         int num_samples)
{
    arena_enter(arena);

    double *errors = arena_push_count(arena, double, num_samples);
    double *dts = arena_push_count(arena, double, num_samples);

    field_size size = { 64, 64, 64 };
    SET_DX(1e-6);

    double T = 1.0;
    double dt = 0.5;

    for (int i = 0; i < num_samples; ++i) {
        arena_enter(arena);

        SET_DT(dt);

        field gamma = field_alloc(size, arena);
        field_fill(size, _DT / (2 * _DX * _DX), gamma);

        field3 solution = field3_alloc(size, arena);
        field3_fill(size, 0, solution);

        uint32_t num_timesteps = T / _DT;
        for (uint32_t t = 1; t < num_timesteps + 1; ++t) {
            arena_enter(arena);

            field3 forcing = field3_alloc(size, arena);
            compute_manufactured_forcing_unsteady(size, t, forcing);
            /* Compute rhs. */
            for (uint32_t i = 0; i < size.depth; ++i) {
                for (uint32_t j = 0; j < size.height; ++j) {
                    for (uint32_t k = 0; k < size.width; ++k) {
                        uint64_t idx = size.height * size.width * i +
                                       size.width * j + k;

                        ftype Dxx_x = 0;
                        ftype Dxx_y = 0;
                        ftype Dxx_z = 0;

                        if (k > 0 && k < size.width - 1) {
                            Dxx_x = (solution.x[idx - 1] -
                                     2 * solution.x[idx] +
                                     solution.x[idx + 1]) / (_DX * _DX);

                            Dxx_y = (solution.y[idx - 1] -
                                     2 * solution.y[idx] +
                                     solution.y[idx + 1]) / (_DX * _DX);

                            Dxx_z = (solution.z[idx - 1] -
                                     2 * solution.z[idx] +
                                     solution.z[idx + 1]) / (_DX * _DX);
                        } else if (k == size.width - 1) {
                            Dxx_y = (4.0 / 3.0 * solution.y[idx - 1] -
                                     4 * solution.y[idx] +
                                     8.0 / 3.0 * get_man_u_y(k * _DX + _DX / 2,
                                                     j * _DX + _DX / 2,
                                                     i * _DX, (t - 1) * _DT)) /
                                                     (_DX * _DX);

                            Dxx_z = (4.0 / 3.0 * solution.z[idx - 1] -
                                     4 * solution.z[idx] +
                                     8.0 / 3.0 * get_man_u_z(k * _DX + _DX / 2,
                                                     j * _DX,
                                                     i * _DX + _DX / 2, (t - 1) * _DT))
                                                     / (_DX * _DX);
                        }

                        forcing.x[idx] = _DT * (forcing.x[idx] + Dxx_x);
                        forcing.y[idx] = _DT * (forcing.y[idx] + Dxx_y);
                        forcing.z[idx] = _DT * (forcing.z[idx] + Dxx_z);
                    }
                }

            }

            field tmp = field_alloc(size, arena);
            field3 solution_delta = field3_alloc(size, arena);
            solve_Dxx_blocks(gamma, size.depth, size.height, size.width,
                             t, tmp, forcing.x, forcing.y, forcing.z,
                             solution_delta.x, solution_delta.y, solution_delta.z);

            for (uint32_t i = 0; i < size.depth; ++i) {
                for (uint32_t j = 0; j < size.height; ++j) {
                    for (uint32_t k = 0; k < size.width; ++k) {
                        uint64_t idx = size.height * size.width * i +
                                       size.width * j + k;

                        solution.x[idx] += solution_delta.x[idx];
                        solution.y[idx] += solution_delta.y[idx];
                        solution.z[idx] += solution_delta.z[idx];
                    }
                }
            }

            arena_exit(arena);
        }

        field3 manufactured = field3_alloc(size, arena);
        compute_manufactured_solution(size, num_timesteps, manufactured);
        errors[i] = field3_l2_norm_diff(size,
                                        _DX,
                                        to_const_field3(solution),
                                        to_const_field3(manufactured));
        dts[i] = _DT;
        dt /= 2;

        arena_exit(arena);
    }

    double *orders = arena_push_count(arena, double, num_samples);
    estimate_convergence_order(errors, dts, num_samples, orders);

    printf("%.8f %e  --\n", dts[0], errors[0]);
    for (int i = 1; i < num_samples; ++i) {
        printf("%.8f %e %5.2f\n", dts[i], errors[i], orders[i]);
        EXPECT_EQUALF(orders[i], 2.0, 0.1);
    }

    arena_exit(arena);
}

int main(void)
{
    ArenaAllocator arena;
    arena_init(&arena, 1ul << 32);

    RUN_TEST(test_convergence_space_momentum_solver_Dxx, &arena, 16, 16, 16, 4);
    RUN_TEST(test_convergence_space_momentum_solver_Dyy, &arena, 16, 16, 16, 4);
    RUN_TEST(test_convergence_space_momentum_solver_Dzz, &arena, 16, 16, 16, 4);

    RUN_TEST(test_convergence_time_momentum_solver_Dxx, &arena, 6);

    arena_destroy(&arena);
}
