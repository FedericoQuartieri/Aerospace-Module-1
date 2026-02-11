#include <stdio.h>
#include <math.h>

#include "test.h"
#include "ftype.h"
#include "field.h"
#include "boundary.h"
#include "consts.h"
#include "solver.h"
#include "convergence-test.h"

//DEFINE_FORCING_TERM()

DEFINE_NU(1.0)
DEFINE_DX(1.0)
DEFINE_DT(1.0)

static const ftype _K = 1.0;

static ftype get_man_u_x(ftype x, ftype y, ftype z, ftype t)
{
    return sin(x) * cos(y + t) * sin(z);
}

static ftype get_man_u_y(ftype x, ftype y, ftype z, ftype t)
{
    return cos(x) * sin(y + t) * sin(z);
}

static ftype get_man_u_z(ftype x, ftype y, ftype z, ftype t)
{
    return 2 * cos(x) * cos(y + t) * cos(z);
}

DEFINE_FUNCTION_BC_U(get_man_u_x, get_man_u_y, get_man_u_z, BC_LEFT)
DEFINE_FUNCTION_BC_U(get_man_u_x, get_man_u_y, get_man_u_z, BC_RIGHT)
DEFINE_FUNCTION_BC_U(get_man_u_x, get_man_u_y, get_man_u_z, BC_TOP)
DEFINE_FUNCTION_BC_U(get_man_u_x, get_man_u_y, get_man_u_z, BC_BOTTOM)
DEFINE_FUNCTION_BC_U(get_man_u_x, get_man_u_y, get_man_u_z, BC_FRONT)
DEFINE_FUNCTION_BC_U(get_man_u_x, get_man_u_y, get_man_u_z, BC_BACK)

static void compute_manufactured_velocity(field_size size,
                                          uint32_t timestep,
                                          field3 dst)
{
    double time = _DT * timestep;

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

static void compute_manufactured_pressure(field_size size,
                                          uint32_t timestep,
                                          field dst)
{
    /* WARNING: Pressure is computed at t + dt/2 */
    double time = timestep * _DT;

    for (uint32_t i = 0; i < size.depth; ++i) {
        for (uint32_t j = 0; j < size.height; ++j) {
            for (uint32_t k = 0; k < size.width; ++k) {
                uint64_t idx = size.height * size.width * i +
                               size.width * j + k;

                    dst[idx] = sin(time) * sin(k * _DX) * sin(j * _DX) *
                               sin(i * _DX);
            }
        }
    }
}

DEF_TEST(test_manufactured_convergence_space,
         ArenaAllocator *arena,
         uint32_t min_depth,
         uint32_t min_height,
         uint32_t min_width,
         int num_timesteps,
         int num_samples)
{
    arena_enter(arena);

    double *velocity_errors = arena_push_count(arena, double, num_samples);
    double *pressure_errors = arena_push_count(arena, double, num_samples);
    double *dxs = arena_push_count(arena, double, num_samples);

    SET_DT(1e-6);

    for (int i = 0; i < num_samples; ++i) {
        arena_enter(arena);

        field_size size = { min_width << i,
                            min_height << i,
                            min_depth << i };

        SET_DX(1.0 / size.width);
        dxs[i] = _DX;

        Solver *solver = solver_alloc(size.depth, size.height,
                                      size.width, arena);
        solver_init(solver);

        /* TODO: Set porosity. */

        for (uint32_t t = 1; t < num_timesteps + 1; ++t) {
            solver_step(solver, t);
        }

        field3 ref_velocity = field3_alloc(size, arena);
        field ref_pressure = field_alloc(size, arena);
        compute_manufactured_velocity(size, num_timesteps, ref_velocity);
        compute_manufactured_pressure(size, num_timesteps, ref_pressure);

        velocity_errors[i] = field3_l2_norm_diff(size,
                                            _DX,
                                            to_const_field3(ref_velocity),
                                            solver_get_velocity(solver));

        pressure_errors[i] = field_l2_norm_diff(size,
                                                _DX,
                                                ref_pressure,
                                                solver_get_pressure(solver));

        arena_exit(arena);
    }

    double *velocity_orders = arena_push_count(arena, double, num_samples);
    double *pressure_orders = arena_push_count(arena, double, num_samples);
    estimate_convergence_order(velocity_errors, dxs,
                               num_samples, velocity_orders);
    estimate_convergence_order(pressure_errors, dxs,
                               num_samples, pressure_orders);

    printf("%3d %12g  --  %12g  --\n", min_width,
           velocity_errors[0], pressure_errors[0]);
    for (int i = 1; i < num_samples; ++i) {
        printf("%3d %12g %.2f %12g %.2f\n", min_width << i,
               velocity_errors[i], velocity_orders[i],
               pressure_errors[i], pressure_orders[i]);
        EXPECT_EQUALF(velocity_orders[i], 2.0, 0.1);
        EXPECT_EQUALF(pressure_orders[i], 2.0, 0.1);
    }

    arena_exit(arena);
}

DEF_TEST(test_manufactured_convergence_time,
         ArenaAllocator *arena,
         int num_samples)
{
    arena_enter(arena);

    double *velocity_errors = arena_push_count(arena, double, num_samples);
    double *pressure_errors = arena_push_count(arena, double, num_samples);
    double *dts = arena_push_count(arena, double, num_samples);

    field_size size = { 64, 64, 64 };
    SET_DX(1.0 / size.width);

    double T = 1.0;
    double dt = 0.1;

    Solver *solver = solver_alloc(size.depth, size.height,
                                  size.width, arena);

    for (int i = 0; i < num_samples; ++i) {
        arena_enter(arena);

        SET_DT(dt);
        dts[i] = _DT;

        solver_init(solver);

        /* TODO: Set porosity. */

        uint32_t num_timesteps = T / _DT;
        printf("%d %f\n", num_timesteps, _DT);
        for (uint32_t t = 1; t < num_timesteps + 1; ++t) {
            solver_step(solver, t);
        }

        field3 ref_velocity = field3_alloc(size, arena);
        field ref_pressure = field_alloc(size, arena);
        compute_manufactured_velocity(size, num_timesteps, ref_velocity);
        compute_manufactured_pressure(size, num_timesteps, ref_pressure);

        velocity_errors[i] = field3_l2_norm_diff(size,
                                                 _DX,
                                                 to_const_field3(ref_velocity),
                                                 solver_get_velocity(solver));

        pressure_errors[i] = field_l2_norm_diff(size,
                                                _DX,
                                                ref_pressure,
                                                solver_get_pressure(solver));

        dt /= 2;

        arena_exit(arena);
    }

    double *velocity_orders = arena_push_count(arena, double, num_samples);
    double *pressure_orders = arena_push_count(arena, double, num_samples);
    estimate_convergence_order(velocity_errors, dts,
                               num_samples, velocity_orders);
    estimate_convergence_order(pressure_errors, dts,
                               num_samples, pressure_orders);

    printf("%.8f %e  --  %e  --\n", dts[0],
           velocity_errors[0], pressure_errors[0]);
    for (int i = 1; i < num_samples; ++i) {
        printf("%.8f %e %5.2f %e %5.2f\n", dts[i],
               velocity_errors[i], velocity_orders[i],
               pressure_errors[i], pressure_orders[i]);
        EXPECT_EQUALF(velocity_orders[i], 2.0, 0.1);
        EXPECT_EQUALF(pressure_orders[i], 2.0, 0.1);
    }

    arena_exit(arena);
}

int main(void)
{
    ArenaAllocator arena;
    arena_init(&arena, 1ul << 32);

    RUN_TEST(test_manufactured_convergence_space, &arena, 8, 8, 8, 10, 4);
    RUN_TEST(test_manufactured_convergence_time, &arena, 4);

    arena_destroy(&arena);

    return 0;
}
