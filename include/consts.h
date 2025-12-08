#ifndef CONSTS_H
#define CONSTS_H

#include "ftype.h"
#include "field.h"

extern const ftype _NU;
extern const ftype _DT;
extern const ftype _DX;

#define DEFINE_NU(x) const ftype _NU = x;
#define DEFINE_DT(x) const ftype _DT = x;
#define DEFINE_DX(x) const ftype _DX = x;

/* TODO: vector getters? */

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
