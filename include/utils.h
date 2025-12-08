#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>

#include "ftype.h"

inline void const_fmemset(ftype *dst, ftype val, uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i) {
        dst[i] = val;
    }
}

inline void rand_fmemset(ftype *dst, uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i) {
        dst[i] = ((ftype) rand()) / RAND_MAX;
    }
}

#endif
