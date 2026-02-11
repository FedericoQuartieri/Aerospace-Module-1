#ifndef BOUNDARY_H
#define BOUNDARY_H

#include "ftype.h"
#include "consts.h"

#define BC_LEFT left
#define BC_RIGHT right
#define BC_TOP top
#define BC_BOTTOM bottom
#define BC_FRONT front
#define BC_BACK back

#define DECLARE_BC_U(boundary) \
        _DECLARE_BC_U(boundary)

#define DEFINE_CONSTANT_BC_U(u_x, u_y, u_z, boundary) \
        _DEFINE_CONSTANT_BC_U(u_x, u_y, u_z, boundary)

#define DEFINE_FUNCTION_BC_U(func_x, func_y, func_z, boundary) \
        _DEFINE_FUNCTION_BC_U(func_x, func_y, func_z, boundary)

#define _DECLARE_BC_U(boundary)                                        \
void _get_##boundary##_bc_u(uint32_t __attribute__((unused)) x,        \
                            uint32_t __attribute__((unused)) y,        \
                            uint32_t __attribute__((unused)) z,        \
                            uint32_t __attribute__((unused)) t,        \
                            vftype *restrict u_x,                      \
                            vftype *restrict u_y,                      \
                            vftype *restrict u_z);                     \
                                                                       \
void _get_##boundary##_bc_u_delta(uint32_t __attribute__((unused)) x,  \
                                  uint32_t __attribute__((unused)) y,  \
                                  uint32_t __attribute__((unused)) z,  \
                                  uint32_t __attribute__((unused)) t,  \
                                  vftype *restrict u_x,                \
                                  vftype *restrict u_y,                \
                                  vftype *restrict u_z);

#define _DEFINE_CONSTANT_BC_U(ux, uy, uz, boundary)                    \
void _get_##boundary##_bc_u(uint32_t __attribute__((unused)) x,        \
                            uint32_t __attribute__((unused)) y,        \
                            uint32_t __attribute__((unused)) z,        \
                            uint32_t __attribute__((unused)) t,        \
                            vftype *restrict u_x,                      \
                            vftype *restrict u_y,                      \
                            vftype *restrict u_z)                      \
{                                                                      \
    *u_x = vbroadcast(ux);                                             \
    *u_y = vbroadcast(uy);                                             \
    *u_z = vbroadcast(uz);                                             \
}                                                                      \
                                                                       \
void _get_##boundary##_bc_u_delta(uint32_t __attribute__((unused)) x,  \
                                  uint32_t __attribute__((unused)) y,  \
                                  uint32_t __attribute__((unused)) z,  \
                                  uint32_t __attribute__((unused)) t,  \
                                  vftype *restrict u_x,                \
                                  vftype *restrict u_y,                \
                                  vftype *restrict u_z)                \
{                                                                      \
    *u_x = vbroadcast(0);                                              \
    *u_y = vbroadcast(0);                                              \
    *u_z = vbroadcast(0);                                              \
}

#define _DEFINE_FUNCTION_BC_U(func_x, func_y, func_z, boundary)           \
void _get_##boundary##_bc_u(uint32_t __attribute__((unused)) x,           \
                            uint32_t __attribute__((unused)) y,           \
                            uint32_t __attribute__((unused)) z,           \
                            uint32_t __attribute__((unused)) t,           \
                            vftype *restrict u_x,                         \
                            vftype *restrict u_y,                         \
                            vftype *restrict u_z)                         \
{                                                                         \
    _##boundary##_bc(x, y, z, t, u_x, u_y, u_z, func_x, func_y, func_z);  \
}                                                                         \
                                                                          \
void _get_##boundary##_bc_u_delta(uint32_t __attribute__((unused)) x,     \
                                  uint32_t __attribute__((unused)) y,     \
                                  uint32_t __attribute__((unused)) z,     \
                                  uint32_t __attribute__((unused)) t,     \
                                  vftype *restrict u_x,                   \
                                  vftype *restrict u_y,                   \
                                  vftype *restrict u_z)                   \
{                                                                         \
    vftype u_x_prev, u_y_prev, u_z_prev;                                  \
    _##boundary##_bc(x, y, z, t - 1,                                      \
                     &u_x_prev, &u_y_prev, &u_z_prev,                     \
                     func_x, func_y, func_z);                             \
                                                                          \
    _##boundary##_bc(x, y, z, t, u_x, u_y, u_z, func_x, func_y, func_z);  \
                                                                          \
    *u_x = *u_x - u_x_prev;                                               \
    *u_y = *u_y - u_y_prev;                                               \
    *u_z = *u_z - u_z_prev;                                               \
}

#define DEFINE_HOMOGENEOUS_BCS_U()           \
    DEFINE_CONSTANT_BC_U(0, 0, 0, BC_LEFT)   \
    DEFINE_CONSTANT_BC_U(0, 0, 0, BC_RIGHT)  \
    DEFINE_CONSTANT_BC_U(0, 0, 0, BC_TOP)    \
    DEFINE_CONSTANT_BC_U(0, 0, 0, BC_BOTTOM) \
    DEFINE_CONSTANT_BC_U(0, 0, 0, BC_FRONT)  \
    DEFINE_CONSTANT_BC_U(0, 0, 0, BC_BACK)

typedef ftype (*bc_func)(ftype x, ftype y, ftype z, ftype t);

static inline __attribute__((always_inline))
void _left_bc(uint32_t x,
              uint32_t y,
              uint32_t z,
              uint32_t t,
              vftype *restrict u_x,
              vftype *restrict u_y,
              vftype *restrict u_z,
              /* These should be inlined. */
              bc_func func_x,
              bc_func func_y,
              bc_func func_z)
{
    /* On the left boundary, we are vectorizing across rows. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    /* This loop should be vectorized. */
    for (int i = 0; i < VLEN; ++i) {
        /* y and z are on the wall */
        tmp_y[i] = func_y(0, (y + i) * _DX + _DX / 2, z * _DX, t * _DT);
        tmp_z[i] = func_z(0, (y + i) * _DX, z * _DX + _DX / 2, t * _DT);

        /* u_1/2 = u0 + dux/dx DX/2 = u0 - (duy/dy + duz/dz) DX/2 */

        ftype duy_dy =
            func_y(0, (y + i) * _DX + _DX / 2, z * _DX, t * _DT) -
            func_y(0, (y + i) * _DX - _DX / 2, z * _DX, t * _DT);

        ftype duz_dz =
            func_z(0, (y + i) * _DX, z * _DX + _DX / 2, t * _DT) -
            func_z(0, (y + i) * _DX, z * _DX - _DX / 2, t * _DT);

        tmp_x[i] = func_x(0, (y + i) * _DX, z * _DX, t * _DT) -
                   (duy_dy + duz_dz) / 2; // Already multiplied by _DX
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline __attribute__((always_inline))
void _top_bc(uint32_t x,
             uint32_t y,
             uint32_t z,
             uint32_t t,
             vftype *restrict u_x,
             vftype *restrict u_y,
             vftype *restrict u_z,
             bc_func func_x,
             bc_func func_y,
             bc_func func_z)
{
    /* On the top boundary, we are vectorizing across columns. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        tmp_x[i] = func_x((x + i) * _DX + _DX / 2, 0, z * _DX, t * _DT);
        tmp_z[i] = func_z((x + i) * _DX, 0, z * _DX + _DX / 2, t * _DT);

        ftype dux_dx =
            (func_x((x + i) * _DX + _DX / 2, 0, z * _DX, t * _DT) -
             func_x((x + i) * _DX - _DX / 2, 0, z * _DX, t * _DT));

        ftype duz_dz =
            (func_z((x + i) * _DX, 0, z * _DX + _DX / 2, t * _DT) -
             func_z((x + i) * _DX, 0, z * _DX - _DX / 2, t * _DT));

        tmp_y[i] = func_y((x + i) * _DX, 0, z * _DX, t * _DT) -
                   (dux_dx + duz_dz) / 2;
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline __attribute__((always_inline))
void _front_bc(uint32_t x,
               uint32_t y,
               uint32_t z,
               uint32_t t,
               vftype *restrict u_x,
               vftype *restrict u_y,
               vftype *restrict u_z,
               bc_func func_x,
               bc_func func_y,
               bc_func func_z)
{
    /* On the front boundary, we are vectorizing across columns. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        tmp_x[i] = func_x((x + i) * _DX + _DX / 2, y * _DX, 0, t * _DT);
        tmp_y[i] = func_y((x + i) * _DX, y * _DX + _DX / 2, 0, t * _DT);

        ftype dux_dx =
            (func_x((x + i) * _DX + _DX / 2, y * _DX, 0, t * _DT) -
             func_x((x + i) * _DX - _DX / 2, y * _DX, 0, t * _DT));

        ftype duy_dy =
            (func_y((x + i) * _DX, y * _DX + _DX / 2, 0, t * _DT) -
             func_y((x + i) * _DX, y * _DX - _DX / 2, 0, t * _DT));

        tmp_z[i] = func_z((x + i) * _DX, y * _DX, 0, t * _DT) -
                   (dux_dx + duy_dy) / 2; /* Already multiplied by _DX */
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline __attribute__((always_inline))
void _right_bc(uint32_t x,
               uint32_t y,
               uint32_t z,
               uint32_t t,
               vftype *restrict u_x,
               vftype *restrict u_y,
               vftype *restrict u_z,
               bc_func func_x,
               bc_func func_y,
               bc_func func_z)
{
    /* On the right boundary, we are vectorizing across rows. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        tmp_x[i] = func_x(
            x * _DX + _DX / 2, (y + i) * _DX, z * _DX, t * _DT);
        tmp_y[i] = func_y(
            x * _DX + _DX / 2, (y + i) * _DX + _DX / 2, z * _DX, t * _DT);
        tmp_z[i] = func_z(
            x * _DX + _DX / 2, (y + i) * _DX, z * _DX + _DX / 2, t * _DT);
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline __attribute__((always_inline))
void _bottom_bc(uint32_t x,
                uint32_t y,
                uint32_t z,
                uint32_t t,
                vftype *restrict u_x,
                vftype *restrict u_y,
                vftype *restrict u_z,
                bc_func func_x,
                bc_func func_y,
                bc_func func_z)
{
    /* On the bottom boundary, we are vectorizing across columns. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        tmp_x[i] = func_x(
            (x + i) * _DX + _DX / 2, y * _DX + _DX / 2, z * _DX, t * _DT);
        tmp_y[i] = func_y(
            (x + i) * _DX, y * _DX + _DX / 2, z * _DX, t * _DT);
        tmp_z[i] = func_z(
            (x + i) * _DX, y * _DX + _DX / 2, z * _DX + _DX / 2, t * _DT);
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

static inline __attribute__((always_inline))
void _back_bc(uint32_t x,
              uint32_t y,
              uint32_t z,
              uint32_t t,
              vftype *restrict u_x,
              vftype *restrict u_y,
              vftype *restrict u_z,
              bc_func func_x,
              bc_func func_y,
              bc_func func_z)
{
    /* On the back boundary, we are vectorizing across columns. */

    ftype __attribute__((aligned(32))) tmp_x[VLEN];
    ftype __attribute__((aligned(32))) tmp_y[VLEN];
    ftype __attribute__((aligned(32))) tmp_z[VLEN];

    for (int i = 0; i < VLEN; ++i) {
        tmp_x[i] = func_x(
            (x + i) * _DX + _DX / 2, y * _DX, z * _DX + _DX / 2, t * _DT);
        tmp_y[i] = func_y(
            (x + i) * _DX, y * _DX + _DX / 2, z * _DX + _DX / 2, t * _DT);
        tmp_z[i] = func_z(
            (x + i) * _DX, y * _DX, z * _DX + _DX / 2, t * _DT);
    }

    *u_x = vload(tmp_x);
    *u_y = vload(tmp_y);
    *u_z = vload(tmp_z);
}

#endif
