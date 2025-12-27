#include <math.h>

#include "test.h"
#include "ftype.h"
#include "alloc.h"
#include "field.h"
#include "consts.h"
#include "convergence-test.h"

#include "pressure.c"

DEFINE_NU(1.0)
DEFINE_DX(1.0)
DEFINE_DT(1.0)

DEF_TEST(test_convergence_pressure_solver_Dxx,
         ArenaAllocator *arena,
         int min_depth,
         int min_height,
         int min_width,
         int num_samples)
{
    arena_enter(arena);

    /* Solve (I - Dxx) p = div(f) with homogeneous neumann BCs */
    /* Choose a solution p that has zero gradient on the boundaries. */

    /*
    *  p = (cosx - 2) siny sinz
    *  px = -sinx siny sinz -> px = 0 for x=0 or x=pi
    *  pxx = -cosx siny sinz
    *
    *  (cosx - 2 + cosx) siny sinz = -div(f)/dt
    *  (2cosx - 2) siny sinz = -div(f)/dt
    *  (2 - 2cosx) siny sinz = div(f)/dt
    *
    *  f.x = -2sinx siny sinz dt
    *  f.y = -cosy sinz
    *  f.z = -siny cosz
    *
    */

    double *errors = arena_push_count(arena, double, num_samples);
    double *dxs = arena_push_count(arena, double, num_samples);

    for (int i = 0; i < num_samples; ++i) {
        arena_enter(arena);

        field_size size = { min_width << i,
                            min_height << i,
                            min_depth << i };

        /* |   |   |   | |
        *  
        * dx * (width - 1) + dx/2 = M_PI
        *
        * dx (width - 1 + 1/2) = M_PI
        *
        * dx = M_PI / (width - 1/2) */

        SET_DX(M_PI / (size.width - 0.5));

        field tmp = field_alloc(size, arena);
        field solution = field_alloc(size, arena);
        field manufactured = field_alloc(size, arena);
        field3 forcing = field3_alloc(size, arena);

        for (uint32_t i = 0; i < size.depth; ++i) {
            for (uint32_t j = 0; j < size.height; ++j) {
                for (uint32_t k = 0; k < size.width; ++k) {
                    uint64_t idx = size.height * size.width * i +
                                   size.width * j + k;

                    manufactured[idx] = (cos(k * _DX) - 2) *
                                        sin(j * _DX) * sin(i * _DX);

                    forcing.x[idx] = -2 * sin(k * _DX + _DX / 2) *
                                     sin(j * _DX) * sin(i * _DX);
                    forcing.y[idx] = -cos(j * _DX + _DX / 2) * sin(i * _DX);
                    forcing.z[idx] = -sin(j * _DX) * cos(i * _DX + _DX / 2);

                }
            }
        }

        solve_Dxx_blocks(forcing.x, forcing.y, forcing.z,
                         size.depth, size.height, size.width, tmp, solution);

        dxs[i] = _DX;
        errors[i] = field_l2_norm_diff(size, _DX, solution, manufactured);

        arena_exit(arena);
    }

    double *orders = arena_push_count(arena, double, num_samples);
    estimate_convergence_order(errors, dxs, num_samples, orders);

    printf("solve_Dxx_blocks()\n");
    print_convergence_table(errors, dxs, orders, num_samples);

    for (int i = 1; i < num_samples; ++i) {
        EXPECT_EQUALF(orders[i], 2.0, 0.05);
    }

    arena_exit(arena);
}

DEF_TEST(test_convergence_pressure_solver_Dyy,
         ArenaAllocator *arena,
         int min_depth,
         int min_height,
         int min_width,
         int num_samples)
{
    arena_enter(arena);

    double *errors = arena_push_count(arena, double, num_samples);
    double *dxs = arena_push_count(arena, double, num_samples);

    for (int i = 0; i < num_samples; ++i) {
        arena_enter(arena);

        field_size size = { min_width << i,
                            min_height << i,
                            min_depth << i };

        SET_DX(M_PI / (size.width - 0.5));

        field tmp = field_alloc(size, arena);
        field solution = field_alloc(size, arena);
        field manufactured = field_alloc(size, arena);
        field forcing = field_alloc(size, arena);

        for (uint32_t i = 0; i < size.depth; ++i) {
            for (uint32_t j = 0; j < size.height; ++j) {
                for (uint32_t k = 0; k < size.width; ++k) {
                    uint64_t idx = size.height * size.width * i +
                                   size.width * j + k;

                    manufactured[idx] = sin(k * _DX) *
                                        cos(j * _DX) *
                                        sin(i * _DX);

                    forcing[idx] = 2 * sin(k * _DX)
                                     * cos(j * _DX)
                                     * sin(i * _DX);
                }
            }
        }

        solve_Dyy_blocks(size.depth, size.height,
                         size.width, tmp, forcing, solution);

        dxs[i] = _DX;
        errors[i] = field_l2_norm_diff(size, _DX, solution, manufactured);

        arena_exit(arena);
    }

    double *orders = arena_push_count(arena, double, num_samples);
    estimate_convergence_order(errors, dxs, num_samples, orders);

    printf("solve_Dyy_blocks()\n");
    print_convergence_table(errors, dxs, orders, num_samples);

    for (int i = 1; i < num_samples; ++i) {
        EXPECT_EQUALF(orders[i], 2.0, 0.05);
    }

    arena_exit(arena);
}

DEF_TEST(test_convergence_pressure_solver_Dzz,
         ArenaAllocator *arena,
         int min_depth,
         int min_height,
         int min_width,
         int num_samples)
{
    arena_enter(arena);

    double *errors = arena_push_count(arena, double, num_samples);
    double *dxs = arena_push_count(arena, double, num_samples);

    for (int i = 0; i < num_samples; ++i) {
        arena_enter(arena);

        field_size size = { min_width << i,
                            min_height << i,
                            min_depth << i };

        SET_DX(M_PI / (size.width - 0.5));

        field tmp = field_alloc(size, arena);
        field solution = field_alloc(size, arena);
        field manufactured = field_alloc(size, arena);
        field forcing = field_alloc(size, arena);

        for (uint32_t i = 0; i < size.depth; ++i) {
            for (uint32_t j = 0; j < size.height; ++j) {
                for (uint32_t k = 0; k < size.width; ++k) {
                    uint64_t idx = size.height * size.width * i +
                                   size.width * j + k;

                    manufactured[idx] = sin(k * _DX) *
                                        sin(j * _DX) *
                                        cos(i * _DX);

                    forcing[idx] = 2 * sin(k * _DX)
                                     * sin(j * _DX)
                                     * cos(i * _DX);
                }
            }
        }

        solve_Dzz_blocks(size.depth, size.height,
                         size.width, tmp, forcing, solution);

        dxs[i] = _DX;
        errors[i] = field_l2_norm_diff(size, _DX, solution, manufactured);

        arena_exit(arena);
    }

    double *orders = arena_push_count(arena, double, num_samples);
    estimate_convergence_order(errors, dxs, num_samples, orders);

    printf("solve_Dzz_blocks()\n");
    print_convergence_table(errors, dxs, orders, num_samples);

    for (int i = 1; i < num_samples; ++i) {
        EXPECT_EQUALF(orders[i], 2.0, 0.05);
    }

    arena_exit(arena);
}

int main(void)
{
    ArenaAllocator arena;
    arena_init(&arena, 1ul << 32);

    RUN_TEST(test_convergence_pressure_solver_Dxx, &arena, 16, 16, 16, 4);
    RUN_TEST(test_convergence_pressure_solver_Dyy, &arena, 16, 16, 16, 4);
    RUN_TEST(test_convergence_pressure_solver_Dzz, &arena, 16, 16, 16, 4);

    arena_destroy(&arena);
}
