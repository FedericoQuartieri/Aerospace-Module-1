#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include "ftype.h"

extern inline void rand_fill(ftype *dst, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        dst[i] = ((ftype) rand()) / RAND_MAX;
    }
}

#endif
