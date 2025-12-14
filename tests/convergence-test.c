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

DEFINE_HOMOGENEOUS_BCS_U()


static double field_l2_dist(field_size size, const_field f1, const_field f2)
{
    double dist = 0;

    uint64_t num_points = field_num_points(size);
    for (uint64_t i = 0; i < num_points; ++i) {
        dist += POW2(f1[i] - f2[i]);
    }

    return sqrt(dist);
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

                printf("%f ", f1.z[idx]);
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

                printf("%f ", f2.z[idx]);
            }
        }
    }
    */

    //printf("\n\nsolution error:");

    for (uint32_t i = 0; i < size.depth;  ++i) {
        //printf("\n");
        for (uint32_t j = 0; j < size.height; ++j) {
            //printf("\n");
            for (uint32_t k = 0; k < size.width; ++k) {

                uint64_t idx = size.height * size.width * i +
                               size.width * j + k;

                /* WARNING: You need to multiply by dx^3!
                 * You're computing a norm! */

                double err = (POW2(f1.x[idx] - f2.x[idx]) +
                              0 * POW2(f1.y[idx] - f2.y[idx]) +
                              0 * POW2(f1.z[idx] - f2.z[idx])) *
                             (_DX * _DX * _DX);

                //printf("%f ", f1.z[idx] - f2.z[idx]);

                dist += err;
            }
        }
    }

    return sqrt(dist);
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

                dst[idx] = -3.0 * _NU * sin(time) * cos(k * _DX)
                                      * sin(j * _DX)
                                      * (sin(i * _DX) - cos(i * _DX));
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

    SET_DT(0.0001);

    for (int i = 0; i < num_samples; ++i) {
        arena_enter(arena);

        field_size size = { max_width >> i,
                            max_height >> i,
                            max_depth >> i };

        SET_DX(1.0 / size.depth);

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
        printf("%3d u=%g p=%g\n", max_depth >> i,
               velocity_errors[i], pressure_errors[i]);
    }

    FAIL_TEST();

    arena_exit(arena);
}

int main(void)
{
    ArenaAllocator arena;
    arena_init(&arena, 1ul << 32);

    RUN_TEST(test_manufactured_convergence_space, &arena, 256, 256, 256, 5);

    arena_destroy(&arena);

    return 0;
}
