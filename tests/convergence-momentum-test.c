#include <math.h>

#include "test.h"
#include "ftype.h"
#include "alloc.h"
#include "field.h"
#include "consts.h"
#include "output.h"
#include "pressure.h"
#include "convergence-test.h"

#include "momentum.c"

DEFINE_DX(1.0)
DEFINE_DT(1.0)
DEFINE_NU(1.0)

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

static ftype get_forcing_x(ftype x, ftype y, ftype z, ftype t)
{
    /* WARNING: _K here is constant! */

    return sin(x) * (-sin(y + t) * sin(z) +
                      cos(y + t) * sin(z) * _NU * (3 + 1.0 / _K) +
                      cos(y + t) * cos(z) * -3 * _NU);
}

static ftype get_forcing_y(ftype x, ftype y, ftype z, ftype t)
{
    return cos(x) * (cos(y + t) * sin(z) +
                     sin(y + t) * sin(z) * _NU * (3 + 1.0 / _K) +
                     sin(y + t) * cos(z) * -3 * _NU);
}

static ftype get_forcing_z(ftype x, ftype y, ftype z, ftype t)
{
    return cos(x) * (sin(y + t) * cos(z) * -2 +
                     cos(y + t) * cos(z) * 2 * _NU * (3 + 1.0 / _K) +
                     cos(y + t) * sin(z) * -3 * _NU);
}

DEFINE_FORCING(get_forcing_x, get_forcing_y, get_forcing_z)

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

                dst.x[idx] =
                    get_man_u_x(_DX * k + _DX / 2, _DX * j, _DX * i, time);
                dst.y[idx] =
                    get_man_u_y(_DX * k, _DX * j + _DX / 2, _DX * i, time);
                dst.z[idx] =
                    get_man_u_z(_DX * k, _DX * j, _DX * i + _DX / 2, time);
            }
        }
    }
}

static void compute_manufactured_pressure(field_size size,
                                          ftype timestep,
                                          field dst)
{
    ftype t = timestep * _DT - _DT / 2;

    for (uint32_t i = 0; i < size.depth; ++i) {
        for (uint32_t j = 0; j < size.height; ++j) {
            for (uint32_t k = 0; k < size.width; ++k) {
                uint64_t idx = size.height * size.width * i +
                               size.width * j + k;

                dst[idx] = 3 * _NU * cos(k * _DX)
                                   * cos(j * _DX + t)
                                   * cos(i * _DX);
            }
        }
    }
}

DEF_TEST(test_convergence_time_splitting_brinkman_auteri,
         ArenaAllocator *arena,
         int num_samples)
{
    arena_enter(arena);

    double *v_errors = arena_push_count(arena, double, num_samples);
    double *p_errors = arena_push_count(arena, double, num_samples);
    double *dts = arena_push_count(arena, double, num_samples);

    double T = 1.0;
    double dt = 0.1;

    field_size size = { 64, 64, 64 };
    SET_DX(M_PI / (size.width - 0.5));

    for (int i = 0; i < num_samples; ++i) {
        arena_enter(arena);

        SET_DT(dt);

        field porosity = field_alloc(size, arena);
        field_fill(size, _K, porosity);

        field gamma = field_alloc(size, arena);
        field_fill(size, (_K * _NU * _DT) /
                         ((2.0 * _K + _NU * _DT) * _DX * _DX), gamma);

        field3 eta = field3_alloc_pad(size, arena);
        field3 zeta = field3_alloc_pad(size, arena);
        field3 vel = field3_alloc_pad(size, arena);

        field3_fill(size, 0, eta);
        field3_fill(size, 0, zeta);
        field3_fill(size, 0, vel);

        field pressure = field_alloc(size, arena);
        field phi = field_alloc(size, arena);

        compute_manufactured_solution(size, 0, eta);
        compute_manufactured_solution(size, 0, zeta);
        compute_manufactured_solution(size, 0, vel);
        compute_manufactured_pressure(size, 0.5, pressure);
        field_fill(size, 0, phi);

        field manufactured_pressure = field_alloc(size, arena);
        compute_manufactured_pressure(size, 0.5, manufactured_pressure);

        /*
        OutputVTK *output = output_vtk_create(size, _DX, arena);

        output_vtk_attach_field(output, pressure, "pressure", arena);
        output_vtk_attach_field(output, manufactured_pressure,
                                "man_pressure", arena);
        output_vtk_attach_field3(output, to_const_field3(vel),
                                 "velocity", arena);

        char output_file_name[32];
        sprintf(output_file_name, "solution-%.4f-%d.vtk", _DT, 0);
        output_vtk_write(output, output_file_name);
        */

        printf("%d/%d\n", i + 1, num_samples);

        uint32_t num_timesteps = round(T / _DT);
        for (uint32_t t = 1; t < num_timesteps + 1; ++t) {
            arena_enter(arena);

            field_size tmp_size = { size.width,
                                    size.height,
                                    size.depth * 4 + 3 };

            field tmp = field_alloc(tmp_size, arena);

            solve_Dxx_blocks_fused_rhs(porosity, gamma, pressure, phi,
                                       eta.x, eta.y, eta.z, zeta.x, zeta.y,
                                       zeta.z, vel.x, vel.y, vel.z,
                                       size.depth, size.height, size.width,
                                       t, tmp);

            solve_Dyy_blocks(gamma, size.depth, size.height, size.width,
                             t, tmp, eta.x, eta.y, eta.z,
                             zeta.x, zeta.y, zeta.z);

            solve_Dzz_blocks(gamma, size.depth, size.height, size.width,
                             t, tmp, zeta.x, zeta.y, zeta.z,
                             vel.x, vel.y, vel.z);

            /*
            field3 vel_old = field3_alloc(size, arena);

            for (uint32_t i = 0; i < size.depth; ++i) {
                for (uint32_t j = 0; j < size.height; ++j) {
                    for (uint32_t k = 0; k < size.width; ++k) {
                        uint64_t idx = size.height * size.width * i +
                                       size.width * j + k;

                        vel_old.x[idx] = vel.x[idx];
                        vel_old.y[idx] = vel.y[idx];
                        vel_old.z[idx] = vel.z[idx];

                    }
                }
            }
            */

            arena_exit(arena);

            /* Now solve pressure. */
            pressure_solve(to_const_field3(vel), size, pressure, phi, t, arena);

            /*
            pressure_correct_rot(to_const_field3(vel),
                                 to_const_field3(vel_old),
                                 size, pressure, 0.0, t);
            */

            /*
            char output_file_name[32];
            sprintf(output_file_name, "solution-%.4f-%d.vtk", _DT, t);
            output_vtk_write(output, output_file_name);
            */
        }

        field3 manufactured_v = field3_alloc(size, arena);
        compute_manufactured_solution(size, num_timesteps, manufactured_v);
        v_errors[i] = field3_l2_norm_diff(size,
                                          _DX,
                                          to_const_field3(vel),
                                          to_const_field3(manufactured_v));

        ftype mean_pressure = 0;
        for (uint64_t i = 0; i < field_num_points(size); ++i) {
            mean_pressure += pressure[i];
        }
        mean_pressure /= field_num_points(size);

        for (uint64_t i = 0; i < field_num_points(size); ++i) {
            pressure[i] -= mean_pressure;
        }


        compute_manufactured_pressure(size, num_timesteps,
                                      manufactured_pressure);
        mean_pressure = 0;
        for (uint64_t i = 0; i < field_num_points(size); ++i) {
            mean_pressure += manufactured_pressure[i];
        }
        mean_pressure /= field_num_points(size);

        for (uint64_t i = 0; i < field_num_points(size); ++i) {
            manufactured_pressure[i] -= mean_pressure;
        }

        p_errors[i] = field_l2_norm_diff(size,
                                         _DX,
                                         pressure,
                                         manufactured_pressure);

        dts[i] = _DT;
        dt /= 2;

        arena_exit(arena);
    }

    double *v_orders = arena_push_count(arena, double, num_samples);
    double *p_orders = arena_push_count(arena, double, num_samples);
    estimate_convergence_order(v_errors, dts, num_samples, v_orders);
    estimate_convergence_order(p_errors, dts, num_samples, p_orders);

    printf("%.8f %e   --  %e   --\n", dts[0], v_errors[0], p_errors[0]);
    for (int i = 1; i < num_samples; ++i) {
        printf("%.8f %e %5.2f %e %5.2f\n",
               dts[i], v_errors[i], v_orders[i], p_errors[i], p_orders[i]);
    }

    arena_exit(arena);
}


int main(void)
{
    ArenaAllocator arena;
    arena_init(&arena, 1ul << 33);

    /*
    RUN_TEST(test_convergence_space_momentum_solver_Dxx, &arena, 16, 16, 16, 5);
    RUN_TEST(test_convergence_space_momentum_solver_Dyy, &arena, 16, 16, 16, 5);
    RUN_TEST(test_convergence_space_momentum_solver_Dzz, &arena, 16, 16, 16, 5);

    RUN_TEST(test_convergence_time_momentum_solver_Dxx, &arena, 5);
    RUN_TEST(test_convergence_time_momentum_solver_Dyy, &arena, 5);

    */
    //RUN_TEST(test_convergence_time_splitting_heat, &arena, 5);
    //RUN_TEST(test_convergence_time_splitting_heat_drag, &arena, 5);
    //RUN_TEST(test_convergence_time_splitting_brinkman, &arena, 5);
    RUN_TEST(test_convergence_time_splitting_brinkman_auteri, &arena, 4);

    arena_destroy(&arena);
}
