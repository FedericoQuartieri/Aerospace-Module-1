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

        /* Enforcing correct value */
        //tmp_x[i] = get_man_u_x(_DX / 2, (y + i) * _DX, z * _DX, t * _DT);

        tmp_x[i] = get_man_u_x(0, (y + i) * _DX, z * _DX, t * _DT) -
                   (duy_dy + duz_dz) / 2; // Already multiplied by _DX

        //tmp_x[i] = get_man_u_x(0, (y + i) * _DX, z * _DX, t * _DT);
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
        //tmp_y[i] = get_man_u_y((x + i) * _DX, _DX / 2, z * _DX, t * _DT);
        //tmp_y[i] = get_man_u_y((x + i) * _DX, 0, z * _DX, t * _DT);
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

        //tmp_z[i] = get_man_u_z((x + i) * _DX, y * _DX, _DX / 2, t * _DT);
        //tmp_z[i] = get_man_u_z((x + i) * _DX, y * _DX, 0, t * _DT);
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

DEFINE_NU(1.0)

static const ftype _K = 1.0;

static void compute_manufactured_forcing_brinkman(field_size size,
                                                  uint32_t timestep,
                                                  field3 dst)
{
    ftype t = timestep * _DT - _DT / 2;

    ftype coeff = _NU * (3 + 1.0 / _K);

    for (uint32_t i = 0; i < size.depth; ++i) {
        for (uint32_t j = 0; j < size.height; ++j) {
            for (uint32_t k = 0; k < size.width; ++k) {
                uint64_t idx = size.height * size.width * i +
                               size.width * j + k;

                dst.x[idx] = sin(k * _DX + _DX / 2) * (-sin(j * _DX + t) * sin(i * _DX) +
                                                       coeff * cos(j * _DX + t) * sin(i * _DX) +
                                                       -3 * _NU * cos(j * _DX + t) * cos(i * _DX));

                dst.y[idx] = cos(k * _DX) * (cos(j * _DX + _DX / 2 + t) * sin(i * _DX) +
                                             coeff * sin(j * _DX + _DX / 2 + t) * sin(i * _DX) +
                                             -3 * _NU * sin(j * _DX + _DX / 2 + t) * cos(i * _DX));

                dst.z[idx] = cos(k * _DX) * (-2 * sin(j * _DX + t) * cos(i * _DX + _DX / 2) +
                                             2 * coeff * cos(j * _DX + t) * cos(i * _DX + _DX / 2) +
                                             -3 * _NU * cos(j * _DX + t) * sin(i * _DX + _DX / 2));
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

        field gamma = field_alloc(size, arena);
        field_fill(size, (_K * _NU * _DT) /
                         ((2.0 * _K + _NU * _DT) * _DX * _DX), gamma);

        ftype beta = (2.0 * _K + _DT * _NU) / (2.0 * _K);

        field3 eps = field3_alloc(size, arena);
        field3 eta = field3_alloc(size, arena);
        field3 zeta = field3_alloc(size, arena);
        field3 vel = field3_alloc(size, arena);

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

            field pressure_pred = field_alloc(size, arena);

            for (uint64_t i = 0; i < field_num_points(size); ++i) {
                pressure_pred[i] = pressure[i] + phi[i];
            }

            field3 forcing = field3_alloc(size, arena);
            compute_manufactured_forcing_brinkman(size, t, forcing);

            /* Compute eps. */
            for (uint32_t i = 0; i < size.depth; ++i) {
                for (uint32_t j = 0; j < size.height; ++j) {
                    for (uint32_t k = 0; k < size.width; ++k) {
                        uint64_t idx = size.height * size.width * i +
                                       size.width * j + k;

                        ftype Dxx_eta_x = 0;
                        ftype Dxx_eta_y = 0;
                        ftype Dxx_eta_z = 0;

                        ftype Dyy_zeta_x = 0;
                        ftype Dyy_zeta_y = 0;
                        ftype Dyy_zeta_z = 0;

                        ftype Dzz_sol_x = 0;
                        ftype Dzz_sol_y = 0;
                        ftype Dzz_sol_z = 0;

                        /* Compute Dxx eta_n */

                        if (k == 0) {
                            /*
                            Dxx_eta_x = (4.0 / 3 * eta.x[idx + 1] -
                                         4 * eta.x[idx] +
                                         8.0 / 3 * get_man_u_x(0, j * _DX, i * _DX,
                                                               (t - 1) * _DT)) / (_DX * _DX);
                            */

                        } else if (k > 0 && k < size.width - 1) {

                            Dxx_eta_x = (eta.x[idx - 1] -
                                         2 * eta.x[idx] +
                                         eta.x[idx + 1]) / (_DX * _DX);

                            Dxx_eta_y = (eta.y[idx - 1] -
                                         2 * eta.y[idx] +
                                         eta.y[idx + 1]) / (_DX * _DX);

                            Dxx_eta_z = (eta.z[idx - 1] -
                                         2 * eta.z[idx] +
                                         eta.z[idx + 1]) / (_DX * _DX);

                        } else if (k == size.width - 1) {

                            /* NOTE: Try to revert to previous approx. */

                            Dxx_eta_y = (4.0 / 3 * eta.y[idx - 1] -
                                         4 * eta.y[idx] +
                                         8.0 / 3 * get_man_u_y(k * _DX + _DX / 2,
                                                               j * _DX + _DX / 2,
                                                               i * _DX, (t - 1) * _DT)) /
                                                               (_DX * _DX);

                            Dxx_eta_z = (4.0 / 3 * eta.z[idx - 1] -
                                         4 * eta.z[idx] +
                                         8.0 / 3 * get_man_u_z(k * _DX + _DX / 2,
                                                               j * _DX,
                                                               i * _DX + _DX / 2,
                                                               (t - 1) * _DT)) / (_DX * _DX);
                        }

                       if (j == 0) {
                            /*
                            Dyy_zeta_y = (4.0 / 3 * zeta.y[idx + size.width] -
                                          4 * zeta.y[idx] +
                                          8.0 / 3 * get_man_u_y(k * _DX, 0, i * _DX,
                                                               (t - 1) * _DT)) / (_DX * _DX);
                            */

                        } else if (j > 0 && j < size.height - 1) {

                            Dyy_zeta_x = (zeta.x[idx - size.width] -
                                          2 * zeta.x[idx] +
                                          zeta.x[idx + size.width]) / (_DX * _DX);

                            Dyy_zeta_y = (zeta.y[idx - size.width] -
                                          2 * zeta.y[idx] +
                                          zeta.y[idx + size.width]) / (_DX * _DX);

                            Dyy_zeta_z = (zeta.z[idx - size.width] -
                                          2 * zeta.z[idx] +
                                          zeta.z[idx + size.width]) / (_DX * _DX);

                        } else if (j == size.height - 1) {

                            Dyy_zeta_x = (4.0 / 3 * zeta.x[idx - size.width] -
                                          4 * zeta.x[idx] +
                                          8.0 / 3 * get_man_u_x(k * _DX + _DX / 2,
                                                                j * _DX + _DX / 2,
                                                                i * _DX, (t - 1) * _DT)) /
                                                                (_DX * _DX);

                            Dyy_zeta_z = (4.0 / 3 * zeta.z[idx - size.width] -
                                          4 * zeta.z[idx] +
                                          8.0 / 3 * get_man_u_z(k * _DX,
                                                                j * _DX + _DX / 2,
                                                                i * _DX + _DX / 2,
                                                                (t - 1) * _DT)) / (_DX * _DX);
                        }

                        if (i == 0) {
                            /*
                            Dzz_sol_z = (4.0 / 3 * vel.z[idx + size.height * size.width] -
                                         4 * vel.z[idx] +
                                         8.0 / 3 * get_man_u_z(k * _DX, j * _DX, 0,
                                                               (t - 1) * _DT)) / (_DX * _DX);
                            */

                        } else if (i > 0 && i < size.depth - 1) {

                            Dzz_sol_x = (vel.x[idx - size.height * size.width] -
                                         2 * vel.x[idx] +
                                         vel.x[idx + size.height * size.width]) / (_DX * _DX);

                            Dzz_sol_y = (vel.y[idx - size.height * size.width] -
                                         2 * vel.y[idx] +
                                         vel.y[idx + size.height * size.width]) / (_DX * _DX);

                            Dzz_sol_z = (vel.z[idx - size.height * size.width] -
                                         2 * vel.z[idx] +
                                         vel.z[idx + size.height * size.width]) / (_DX * _DX);

                        } else if (i == size.depth - 1) {

                            Dzz_sol_x = (4.0 / 3 * vel.x[idx - size.height * size.width] -
                                         4 * vel.x[idx] +
                                         8.0 / 3 * get_man_u_x(k * _DX + _DX / 2,
                                                               j * _DX,
                                                               i * _DX + _DX / 2, (t - 1) * _DT)) /
                                                               (_DX * _DX);

                            Dzz_sol_y = (4.0 / 3 * vel.y[idx - size.height * size.width] -
                                         4 * vel.y[idx] +
                                         8.0 / 3 * get_man_u_y(k * _DX,
                                                               j * _DX + _DX / 2,
                                                               i * _DX + _DX / 2, (t - 1) * _DT)) /
                                                               (_DX * _DX);
                        }

                        ftype Dx_p = 0;
                        ftype Dy_p = 0;
                        ftype Dz_p = 0;

                        if (k < size.width - 1) {
                            Dx_p = (pressure_pred[idx + 1] -
                                    pressure_pred[idx]) / _DX;
                        }

                        if (j < size.height - 1) {
                            Dy_p = (pressure_pred[idx + size.width] -
                                    pressure_pred[idx]) / _DX;
                        }

                        if (i < size.depth - 1) {
                            Dz_p = (pressure_pred[idx + size.height * size.width] -
                                    pressure_pred[idx]) / _DX;
                        }

                        eps.x[idx] = vel.x[idx] + _DT / beta * (forcing.x[idx] +
                            _NU * (Dxx_eta_x +
                                       Dyy_zeta_x +
                                       Dzz_sol_x -
                                       1.0 / _K * vel.x[idx]) -
                            Dx_p);

                        eps.y[idx] = vel.y[idx] + _DT / beta * (forcing.y[idx] +
                            _NU * (Dxx_eta_y +
                                       Dyy_zeta_y +
                                       Dzz_sol_y -
                                       1.0 / _K * vel.y[idx]) -
                            Dy_p);

                        eps.z[idx] = vel.z[idx] + _DT / beta * (forcing.z[idx] +
                            _NU * (Dxx_eta_z +
                                       Dyy_zeta_z +
                                       Dzz_sol_z -
                                       1.0 / _K * vel.z[idx]) -
                            Dz_p);
                    }
                }
            }

            field tmp = field_alloc(size, arena);
            field3 rhs = field3_alloc(size, arena);
            field3 delta = field3_alloc(size, arena);

            /* Solve for eta_n+1 */

            for (uint32_t i = 0; i < size.depth; ++i) {
                for (uint32_t j = 0; j < size.height; ++j) {
                    for (uint32_t k = 0; k < size.width; ++k) {
                        uint64_t idx = size.height * size.width * i +
                                       size.width * j + k;

                        rhs.x[idx] = eps.x[idx] - eta.x[idx];
                        rhs.y[idx] = eps.y[idx] - eta.y[idx];
                        rhs.z[idx] = eps.z[idx] - eta.z[idx];
                    }
                }
            }

            solve_Dxx_blocks(gamma, size.depth, size.height, size.width,
                             t, tmp, rhs.x, rhs.y, rhs.z,
                             delta.x, delta.y, delta.z);

            for (uint32_t i = 0; i < size.depth; ++i) {
                for (uint32_t j = 0; j < size.height; ++j) {
                    for (uint32_t k = 0; k < size.width; ++k) {
                        uint64_t idx = size.height * size.width * i +
                                       size.width * j + k;

                        eta.x[idx] += delta.x[idx];
                        eta.y[idx] += delta.y[idx];
                        eta.z[idx] += delta.z[idx];
                    }
                }
            }

            /* Solve for zeta_n+1 */

            for (uint32_t i = 0; i < size.depth; ++i) {
                for (uint32_t j = 0; j < size.height; ++j) {
                    for (uint32_t k = 0; k < size.width; ++k) {
                        uint64_t idx = size.height * size.width * i +
                                       size.width * j + k;

                        rhs.x[idx] = eta.x[idx] - zeta.x[idx];
                        rhs.y[idx] = eta.y[idx] - zeta.y[idx];
                        rhs.z[idx] = eta.z[idx] - zeta.z[idx];
                    }
                }
            }

            solve_Dyy_blocks(gamma, size.depth, size.height, size.width,
                             t, tmp, rhs.x, rhs.y, rhs.z,
                             delta.x, delta.y, delta.z);

            for (uint32_t i = 0; i < size.depth; ++i) {
                for (uint32_t j = 0; j < size.height; ++j) {
                    for (uint32_t k = 0; k < size.width; ++k) {
                        uint64_t idx = size.height * size.width * i +
                                       size.width * j + k;

                        zeta.x[idx] += delta.x[idx];
                        zeta.y[idx] += delta.y[idx];
                        zeta.z[idx] += delta.z[idx];
                    }
                }
            }

            /* Solve for zeta_n+1 */

            for (uint32_t i = 0; i < size.depth; ++i) {
                for (uint32_t j = 0; j < size.height; ++j) {
                    for (uint32_t k = 0; k < size.width; ++k) {
                        uint64_t idx = size.height * size.width * i +
                                       size.width * j + k;

                        rhs.x[idx] = zeta.x[idx] - vel.x[idx];
                        rhs.y[idx] = zeta.y[idx] - vel.y[idx];
                        rhs.z[idx] = zeta.z[idx] - vel.z[idx];
                    }
                }
            }

            solve_Dzz_blocks(gamma, size.depth, size.height, size.width,
                             t, tmp, rhs.x, rhs.y, rhs.z,
                             delta.x, delta.y, delta.z);

            field3 vel_old = field3_alloc(size, arena);

            for (uint32_t i = 0; i < size.depth; ++i) {
                for (uint32_t j = 0; j < size.height; ++j) {
                    for (uint32_t k = 0; k < size.width; ++k) {
                        uint64_t idx = size.height * size.width * i +
                                       size.width * j + k;

                        vel_old.x[idx] = vel.x[idx];
                        vel_old.y[idx] = vel.y[idx];
                        vel_old.z[idx] = vel.z[idx];

                        vel.x[idx] += delta.x[idx];
                        vel.y[idx] += delta.y[idx];
                        vel.z[idx] += delta.z[idx];
                    }
                }
            }

            arena_exit(arena);

            /* Now solve pressure. */
            pressure_solve(to_const_field3(vel), size, pressure, phi, t, arena);

            pressure_correct_rot(to_const_field3(vel),
                                 to_const_field3(vel_old),
                                 size, pressure, 0.0, t);


            //char output_file_name[32];
            //sprintf(output_file_name, "solution-%.4f-%d.vtk", _DT, t);
            //output_vtk_write(output, output_file_name);
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
