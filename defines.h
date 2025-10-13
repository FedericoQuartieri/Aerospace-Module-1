
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* WARNING: multiple of tile size */
#ifndef WIDTH 
    #define WIDTH 320 
#endif
#ifndef HEIGHT
    #define HEIGHT 320
#endif
#ifndef DEPTH
    #define DEPTH 32
#endif


#ifdef FLOAT
    #ifndef TILE_WIDTH
        #define TILE_WIDTH 8
    #endif
    #ifndef DTYPE
        #define DTYPE float
    #endif
#else
    #ifndef TILE_WIDTH
        #define TILE_WIDTH 4
    #endif
    #ifndef DTYPE
        #define DTYPE double
    #endif
#endif

#ifndef TILE_HEIGHT
    #define TILE_HEIGHT 2
#endif
#ifndef TILE_DEPTH
    #define TILE_DEPTH 2
#endif

#ifndef nu_const
    #define nu_const 0.7
#endif

#ifndef d_x
    #define d_x (1.0/320)
#endif

#ifndef d_x_inverse
    #define d_x_inverse 320
#endif

#ifndef d_x_inverse_squared
    #define d_x_inverse_squared 102400
#endif

#ifndef d_y
    #define d_y (1.0/320)
#endif

#ifndef d_y_inverse
    #define d_y_inverse 320
#endif

#ifndef d_y_inverse_squared
    #define d_y_inverse_squared 102400
#endif

#ifndef d_z
    #define d_z (1.0/32)
#endif

#ifndef d_z_inverse
    #define d_z_inverse 32
#endif

#ifndef d_z_inverse_squared
    #define d_z_inverse_squared 1024
#endif
