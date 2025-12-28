#include <stdio.h>
#include <math.h>

#include "test.h"
#include "ftype.h"
#include "boundary.h"
#include "consts.h"
#include "solver.h"
#include "convergence-test.h"

#define POW2(x) ((x) * (x))

//DEFINE_FORCING_TERM()

DEFINE_NU(1.0)
DEFINE_DX(1.0)
DEFINE_DT(1.0)

static inline ftype get_man_u_x(ftype x, ftype y, ftype z, ftype time)
{
    return sin(time) * sin(x) * sin(y) * sin(z);
}

static inline ftype get_man_u_y(ftype x, ftype y, ftype z, ftype time)
{
    return sin(time) * cos(x) * cos(y) * cos(z);
}

static inline ftype get_man_u_z(ftype x, ftype y, ftype z, ftype time)
{
    return sin(time) * cos(x) * sin(y) * (cos(z) + sin(z));
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

    /* At the moment, work in serial fashion, consider wrapping
     * this behaviour in the macro definition. */
    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    /* WARNING: What happens when y+i-1 < 0 ? Solution should be
     * enforced there, since we are sitting on the top wall. */
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

static double field_l2_dist(field_size size, const_field f1, const_field f2)
{
    double dist = 0;

    /*
    printf("\n\nreference:");

    for (uint32_t i = 0; i < size.depth;  ++i) {
        printf("\n");
        for (uint32_t j = 0; j < size.height; ++j) {
            printf("\n");
            for (uint32_t k = 0; k < size.width; ++k) {

                uint64_t idx = size.height * size.width * i +
                               size.width * j + k;

                printf("%g ", f1[idx]);
            }
        }
    }

    printf("\n\nsolution:");

    for (uint32_t i = 0; i < size.depth;  ++i) {
        printf("\n");
        for (uint32_t j = 0; j < size.height; ++j) {
            printf("\n");
            for (uint32_t k = 0; k < size.width; ++k) {

                uint64_t idx = size.height * size.width * i +
                               size.width * j + k;

                printf("%g ", f2[idx]);
            }
        }
    }

    printf("\n\nsolution error:");
    */

    for (uint32_t i = 0; i < size.depth; ++i) {
        //printf("\n");
        for (uint32_t j = 0; j < size.height; ++j) {
            //printf("\n");
            for (uint32_t k = 0; k < size.width; ++k) {
                uint64_t idx = size.height * size.width * i +
                               size.width * j + k;

                double err = POW2(f1[idx] - f2[idx]);
                dist += err;

                //printf("%g ", err);
            }
        }
    }

    return sqrt(dist * _DX * _DX * _DX);
}

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

                dst.x[idx] = sin(time) * sin(k * _DX + _DX / 2)
                                       * sin(j * _DX)
                                       * sin(i * _DX);

                dst.y[idx] = sin(time) * cos(k * _DX)
                                       * cos(j * _DX + _DX / 2)
                                       * cos(i * _DX);

                dst.z[idx] = sin(time) * cos(k * _DX)
                                       * sin(j * _DX)
                                       * (cos(i * _DX + _DX / 2) +
                                          sin(i * _DX + _DX / 2));
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

                /*
                    dst[idx] = sin(time) * sin(k * _DX) * sin(j * _DX) *
                               sin(i * _DX);
                */
                dst[idx] = 0;
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

        pressure_errors[i] = field_l2_dist(size,
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

        pressure_errors[i] = field_l2_dist(size,
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
