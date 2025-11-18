#ifndef BCS_H
#define BCS_H

#include "ftype.h"

#define BC_LEFT left
#define BC_RIGHT right
#define BC_TOP top
#define BC_BOTTOM bot
#define BC_FRONT front
#define BC_BACK back

#define DEF_CONSTANT_BC_U(u_x, u_y, u_z, boundary)               \
inline __attribute__((always_inline))                            \
void _get_##boundary##_bc_u(uint32_t __attribute__((unused)) x,  \
                            uint32_t __attribute__((unused)) y,  \
                            uint32_t __attribute__((unused)) z,  \
                            vftype *u_x,                         \
                            vftype *u_y,                         \
                            vftype *u_z)                         \
{                                                                \
    *u_x = vbroadcast(u_x);                                      \
    *u_y = vbroadcast(u_x);                                      \
    *u_z = vbroadcast(u_x);                                      \
}

/* TODO: Use LTO to enable inlining across translation units. */
#define DEF_FUNCTION_BC_U(boundary, func)                        \
inline __attribute__((always_inline))                            \
void _get_##boundary##_bc_u(uint32_t __attribute__((unused)) x,  \
                            uint32_t __attribute__((unused)) y,  \
                            uint32_t __attribute__((unused)) z,  \
                            ftype dx,                            \
                            vftype *u_x,                         \
                            vftype *u_y,                         \
                            vftype *u_z)                         \
{                                                                \
    func(x, y, z, dx, u_x, u_y, u_z);                            \
}

#endif
