#ifndef BCS_H
#define BCS_H

#include "ftype.h"

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

#define DEFINE_FUNCTION_BC_U(func, boundary) \
        _DEFINE_FUNCTION_BC_U(func, boundary)

#define _DECLARE_BC_U(boundary)                                  \
void _get_##boundary##_bc_u(uint32_t __attribute__((unused)) x,  \
                            uint32_t __attribute__((unused)) y,  \
                            uint32_t __attribute__((unused)) z,  \
                            vftype *__restrict__ u_x,            \
                            vftype *__restrict__ u_y,            \
                            vftype *__restrict__ u_z);

#define _DEFINE_CONSTANT_BC_U(ux, uy, uz, boundary)              \
void _get_##boundary##_bc_u(uint32_t __attribute__((unused)) x,  \
                            uint32_t __attribute__((unused)) y,  \
                            uint32_t __attribute__((unused)) z,  \
                            vftype *__restrict__ u_x,            \
                            vftype *__restrict__ u_y,            \
                            vftype *__restrict__ u_z)            \
{                                                                \
    *u_x = vbroadcast(ux);                                       \
    *u_y = vbroadcast(uy);                                       \
    *u_z = vbroadcast(uz);                                       \
}

/* TODO: Use LTO to enable inlining across translation units. */
#define _DEFINE_FUNCTION_BC_U(func, boundary)                    \
void _get_##boundary##_bc_u(uint32_t __attribute__((unused)) x,  \
                            uint32_t __attribute__((unused)) y,  \
                            uint32_t __attribute__((unused)) z,  \
                            ftype dx,                            \
                            vftype *__restrict__ u_x,            \
                            vftype *__restrict__ u_y,            \
                            vftype *__restrict__ u_z)            \
{                                                                \
    func(x, y, z, dx, u_x, u_y, u_z);                            \
}

#endif
