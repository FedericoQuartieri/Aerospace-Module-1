#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <math.h>

// ==================== CONFIGURABLE VIA CMAKE ====================
// Grid dimensions (overridable via -DWIDTH_OVERRIDE=N etc.)
#ifdef WIDTH_OVERRIDE
    #define WIDTH WIDTH_OVERRIDE
#else
    #define WIDTH 256
#endif

#ifdef HEIGHT_OVERRIDE
    #define HEIGHT HEIGHT_OVERRIDE
#else
    #define HEIGHT 256
#endif

#ifdef DEPTH_OVERRIDE
    #define DEPTH DEPTH_OVERRIDE
#else
    #define DEPTH 256
#endif

// Grid spatial dimension (number of elements)
#define GRID_ELEMENTS WIDTH * HEIGHT * DEPTH

// Total field size in bytes
#define GRID_SIZE (GRID_ELEMENTS * sizeof(DTYPE))

// Data type
#define DTYPE double

// Physical constants 
#define NU 1.0                // Kinematic viscosity (nu)
// In the adimensional navier-stokes formulation with L=[0,1] U=1, T=L/U=1 then -> 1/Re = NU

// Physical domain
#define LX M_PI  // Domain [0, PI] in x
#define LY M_PI  // Domain [0, PI] in y
#define LZ M_PI  // Domain [0, PI] in z
/* 
    Staggered grid:

    L       /---------------------/
    grid    *-->--*-->--*-->--*-->
    DX/2    /--/ 
    DX      /-----/
    
    If the number of points is WIDTH (4 in this case) then:
    DX/2 = L / (2WIDTH - 1)
    DX = 2L / (2WIDTH - 1)
*/
#define DX ((2 * LX) / (DTYPE)(2*WIDTH - 1))   // Grid spacing in x 
#define DY ((2 * LY) / (DTYPE)(2*HEIGHT - 1))  // Grid spacing in y
#define DZ ((2 * LZ) / (DTYPE)(2*DEPTH - 1))   // Grid spacing in z

// Time stepping (overridable via -DDT_OVERRIDE=0.01 etc.)
#ifdef DT_OVERRIDE
    #define DT DT_OVERRIDE
#else
    #define DT 0.001
#endif

#ifdef TOTAL_TIME_OVERRIDE
    #define TOTAL_TIME TOTAL_TIME_OVERRIDE
#else
    #define TOTAL_TIME 0.005
#endif

#define STEPS ((int)(TOTAL_TIME / DT))  // Number of time steps
#define WRITE_FREQUENCY STEPS/STEPS            // Writing output frequency

#define DX_INVERSE (1.0 / DX)
#define DY_INVERSE (1.0 / DY)
#define DZ_INVERSE (1.0 / DZ)

#define DX_INVERSE_SQUARE (DX_INVERSE * DX_INVERSE)
#define DY_INVERSE_SQUARE (DY_INVERSE * DY_INVERSE)
#define DZ_INVERSE_SQUARE (DZ_INVERSE * DZ_INVERSE)

// Simulation parameters
#define MAX_ITERATIONS 1000
#define TOLERANCE 1e-6

#if defined(USE_DOUBLE)
    #define TYPE double
#elif defined(USE_FLOAT)
    #define TYPE float
#else
    #define USE_FLOAT
#endif

#endif // CONSTANTS_H