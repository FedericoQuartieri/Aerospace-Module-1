#include <stdio.h>
#include <math.h>

#include "test.h"
#include "ftype.h"
#include "boundary.h"
#include "consts.h"
#include "solver.h"

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
        tmp_y[i] = get_man_u_y(0, (y + i) * _DX + _DX / 2, z * _DX, _DT);
        tmp_z[i] = get_man_u_z(0, (y + i) * _DX, z * _DX + _DX / 2, _DT);

        /* u_1/2 = u0 + dux/dx DX/2 = u0 - (duy/dy + duz/dz) DX/2 */

        ftype duy_dy =
            get_man_u_y(0, (y + i) * _DX + _DX / 2, z * _DX, _DT) -
            get_man_u_y(0, (y + i) * _DX - _DX / 2, z * _DX, _DT);

        ftype duz_dz =
            get_man_u_z(0, (y + i) * _DX, z * _DX + _DX / 2, _DT) -
            get_man_u_z(0, (y + i) * _DX, z * _DX - _DX / 2, _DT);

        tmp_x[i] = get_man_u_x(0, (y + i) * _DX, z * _DX, _DT) -
                   (duy_dy + duz_dz) / 2; /* Already multiplied by _DX */
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline void top_bc_manufactured(uint32_t x,
                                       uint32_t y,
                                       uint32_t z,
                                       vftype *restrict u_x,
                                       vftype *restrict u_y,
                                       vftype *restrict u_z)
{
    /* On the top boundary, we are vectorizing across columns. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        tmp_x[i] = get_man_u_x((x + i) * _DX + _DX / 2, 0, z * _DX, _DT);
        tmp_z[i] = get_man_u_z((x + i) * _DX, 0, z * _DX + _DX / 2, _DT);

        ftype dux_dx =
            (get_man_u_x((x + i) * _DX + _DX / 2, 0, z * _DX, _DT) -
             get_man_u_x((x + i) * _DX - _DX / 2, 0, z * _DX, _DT));

        ftype duz_dz =
            (get_man_u_z((x + i) * _DX, 0, z * _DX + _DX / 2, _DT) -
             get_man_u_z((x + i) * _DX, 0, z * _DX - _DX / 2, _DT));

        tmp_y[i] = get_man_u_y((x + i) * _DX, 0, z * _DX, _DT) -
                   (dux_dx + duz_dz) / 2;
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline void front_bc_manufactured(uint32_t x,
                                         uint32_t y,
                                         uint32_t z,
                                         vftype *restrict u_x,
                                         vftype *restrict u_y,
                                         vftype *restrict u_z)
{
    /* On the front boundary, we are vectorizing across columns. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        tmp_x[i] = get_man_u_x((x + i) * _DX + _DX / 2, y * _DX, 0, _DT);
        tmp_y[i] = get_man_u_y((x + i) * _DX, y * _DX + _DX / 2, 0, _DT);

        ftype dux_dx =
            (get_man_u_x((x + i) * _DX + _DX / 2, y * _DX, 0, _DT) -
             get_man_u_x((x + i) * _DX - _DX / 2, y * _DX, 0, _DT));

        ftype duy_dy =
            (get_man_u_y((x + i) * _DX, y * _DX + _DX / 2, 0, _DT) -
             get_man_u_y((x + i) * _DX, y * _DX - _DX / 2, 0, _DT));

        tmp_z[i] = get_man_u_z((x + i) * _DX, y * _DX, 0, _DT) -
                   (dux_dx + duy_dy) / 2; /* Already multiplied by _DX */
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline void right_bc_manufactured(uint32_t x,
                                         uint32_t y,
                                         uint32_t z,
                                         vftype *restrict u_x,
                                         vftype *restrict u_y,
                                         vftype *restrict u_z)
{
    /* On the front boundary, we are vectorizing across columns. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        tmp_x[i] = get_man_u_x(x * _DX + _DX / 2, (y + i) * _DX, z * _DX, _DT);
        tmp_y[i] = get_man_u_y(x * _DX + _DX / 2, (y + i) * _DX + _DX / 2, z * _DX, _DT);
        tmp_z[i] = get_man_u_z(x * _DX + _DX / 2, (y + i) * _DX, z * _DX + _DX / 2, _DT);
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline void bottom_bc_manufactured(uint32_t x,
                                          uint32_t y,
                                          uint32_t z,
                                          vftype *restrict u_x,
                                          vftype *restrict u_y,
                                          vftype *restrict u_z)
{
    /* On the front boundary, we are vectorizing across columns. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        tmp_x[i] = get_man_u_x((x + i) * _DX + _DX / 2, y * _DX + _DX / 2, z * _DX, _DT);
        tmp_y[i] = get_man_u_y((x + i) * _DX, y * _DX + _DX / 2, z * _DX, _DT);
        tmp_z[i] = get_man_u_z((x + i) * _DX, y * _DX + _DX / 2, z * _DX + _DX / 2, _DT);
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline void back_bc_manufactured(uint32_t x,
                                        uint32_t y,
                                        uint32_t z,
                                        vftype *restrict u_x,
                                        vftype *restrict u_y,
                                        vftype *restrict u_z)
{
    /* On the front boundary, we are vectorizing across columns. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        tmp_x[i] = get_man_u_x((x + i) * _DX + _DX / 2, y * _DX, z * _DX + _DX / 2, _DT);
        tmp_y[i] = get_man_u_y((x + i) * _DX, y * _DX + _DX / 2, z * _DX + _DX / 2, _DT);
        tmp_z[i] = get_man_u_z((x + i) * _DX, y * _DX + _DX / 2, z * _DX, _DT);
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

DEFINE_FUNCTION_BC_U(left_bc_manufactured, BC_LEFT)
DEFINE_FUNCTION_BC_U(top_bc_manufactured, BC_TOP)
DEFINE_FUNCTION_BC_U(front_bc_manufactured, BC_FRONT)
DEFINE_FUNCTION_BC_U(right_bc_manufactured, BC_RIGHT)
DEFINE_FUNCTION_BC_U(bottom_bc_manufactured, BC_BOTTOM)
DEFINE_FUNCTION_BC_U(back_bc_manufactured, BC_BACK)

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

static double field3_l2_dist(field_size size, const_field3 f1, const_field3 f2)
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

                printf("%g ", f1.x[idx]);
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

                printf("%g ", f2.x[idx]);
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

                /* WARNING: You need to multiply by dx^3!
                 * You're computing a norm! */

                double err = POW2(f1.x[idx] - f2.x[idx]) +
                             POW2(f1.y[idx] - f2.y[idx]) +
                             POW2(f1.z[idx] - f2.z[idx]);

                //printf("%g ", fabs(((f1.x[idx] - f2.x[idx]) / fmax(f1.x[idx], 1e-12))));

                dist += err;
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
    double time = _DT * timestep;

    for (uint32_t i = 0; i < size.depth; ++i) {
        for (uint32_t j = 0; j < size.height; ++j) {
            for (uint32_t k = 0; k < size.width; ++k) {
                uint64_t idx = size.height * size.width * i +
                               size.width * j + k;

                dst[idx] = 0.0;//-3.0 * _NU * sin(time) * cos(k * _DX)
                               //       * sin(j * _DX)
                               //       * (sin(i * _DX) - cos(i * _DX));
            }
        }
    }
}

DEF_TEST(test_manufactured_convergence_space,
         ArenaAllocator *arena,
         uint32_t max_depth,
         uint32_t max_height,
         uint32_t max_width,
         int num_samples)
{
    arena_enter(arena);

    double *velocity_errors = arena_push_count(arena, double, num_samples);
    double *pressure_errors = arena_push_count(arena, double, num_samples);

    SET_DT(0.000001);

    for (int i = 0; i < num_samples; ++i) {
        arena_enter(arena);

        field_size size = { max_width >> i,
                            max_height >> i,
                            max_depth >> i };

        SET_DX(1.0 / size.width);

        Solver *solver = solver_alloc(size.depth, size.height,
                                      size.width, arena);
        solver_init(solver);

        /* TODO: Set porosity. */

        solver_step(solver, 1);

        field3 ref_velocity = field3_alloc(size, arena);
        field ref_pressure = field_alloc(size, arena);
        compute_manufactured_velocity(size, 1, ref_velocity);
        compute_manufactured_pressure(size, 1, ref_pressure);

        velocity_errors[i] = field3_l2_dist(size,
                                            to_const_field3(ref_velocity),
                                            solver_get_velocity(solver));

        pressure_errors[i] = field_l2_dist(size,
                                           ref_pressure,
                                           solver_get_pressure(solver));

        arena_exit(arena);
    }

    for (int i = 0; i < num_samples; ++i) {
        printf("%3d u=%g p=%g\n", max_width >> i,
               velocity_errors[i], pressure_errors[i]);
    }

    FAIL_TEST();

    arena_exit(arena);
}

int main(void)
{
    ArenaAllocator arena;
    arena_init(&arena, 1ul << 32);

    RUN_TEST(test_manufactured_convergence_space, &arena, 256, 256, 256, 4);

    arena_destroy(&arena);

    return 0;
}
