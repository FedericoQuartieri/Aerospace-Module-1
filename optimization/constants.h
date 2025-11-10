#if defined(USE_DOUBLE)
    #define TYPE double
#elif defined(USE_FLOAT)
    #define TYPE float
#else
    #define USE_FLOAT
    #define TYPE float // default
#endif
#define N 513 // 512 + 1(ghost cell)
#define GRID_ELEMENTS N*N*N
