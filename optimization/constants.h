#ifndef OPTIMIZATION_CONSTANTS_H
#define OPTIMIZATION_CONSTANTS_H

#if defined(USE_DOUBLE)
    #define TYPE double
#elif defined(USE_FLOAT)
    #define TYPE float
#else
    #define USE_DOUBLE
    #define TYPE double // default
#endif
#ifdef N_OVERRIDE
    #define N N_OVERRIDE
#else
    #define N 513 // 512 + 1(ghost cell)
#endif
#define GRID_ELEMENTS ((size_t)N * (size_t)N * (size_t)N)

#endif
