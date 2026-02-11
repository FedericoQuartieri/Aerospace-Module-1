#ifndef CONSTS_H
#define CONSTS_H

#include "ftype.h"
#include "field.h"

extern ftype _NU;
extern ftype _DT;
extern ftype _DX;

#define DEFINE_NU(x) ftype _NU = x;
#define DEFINE_DT(x) ftype _DT = x;
#define DEFINE_DX(x) ftype _DX = x;

#define SET_NU(x) _NU = x
#define SET_DT(x) _DT = x
#define SET_DX(x) _DX = x

/* TODO: vector getters? */

#define DECLARE_FORCING()                              \
void _get_forcing(uint32_t __attribute__((unused)) x,  \
                  uint32_t __attribute__((unused)) y,  \
                  uint32_t __attribute__((unused)) z,  \
                  uint32_t __attribute__((unused)) t,  \
                  vftype *restrict f_x,                \
                  vftype *restrict f_y,                \
                  vftype *restrict f_z);               \

#define DEFINE_FORCING(func)                           \
void _get_forcing(uint32_t __attribute__((unused)) x,  \
                  uint32_t __attribute__((unused)) y,  \
                  uint32_t __attribute__((unused)) z,  \
                  uint32_t __attribute__((unused)) t,  \
                  vftype *restrict f_x,                \
                  vftype *restrict f_y,                \
                  vftype *restrict f_z)                \
{                                                      \
    func(x, y, z, t, f_x, f_y, f_z);                   \
}

#define DEFINE_CONSTANT_FORCING(fx, fy, fz)            \
void _get_forcing(uint32_t __attribute__((unused)) x,  \
                  uint32_t __attribute__((unused)) y,  \
                  uint32_t __attribute__((unused)) z,  \
                  uint32_t __attribute__((unused)) t,  \
                  vftype *restrict f_x,                \
                  vftype *restrict f_y,                \
                  vftype *restrict f_z)                \
{                                                      \
    *f_x = vbroadcast(fx);                             \
    *f_y = vbroadcast(fy);                             \
    *f_z = vbroadcast(fz);                             \
}

static inline void compute_gamma(const_field porosity,
                                 field_size size,
                                 field dst)
{
    uint64_t num_points = field_num_points(size);
    for (uint64_t i = 0; i < num_points; ++i) {
        ftype k = porosity[i];
        dst[i] = (k * _DT * _NU) / (2 * k + _DT * _NU) / (_DX * _DX);
    }
}

#endif
