#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdlib.h>

#include "ftype.h"

inline void rand_fmemset(ftype *dst, uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i) {
        dst[i] = ((ftype) rand()) / RAND_MAX;
    }
}

inline void fmemset(ftype *dst, ftype x, uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i) {
        dst[i] = x;
    }

}

#endif
