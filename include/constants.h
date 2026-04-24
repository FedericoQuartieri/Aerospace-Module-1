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

// Data precision type
#if !defined(USE_FLOAT) && !defined(USE_DOUBLE)
    #define USE_DOUBLE
#endif

#if defined(USE_DOUBLE)
    #define DTYPE double
#elif defined(USE_FLOAT)
    #define DTYPE float
#endif

// Time measurement
#define START(name) uint64_t name##_start = time_ns()
#define END_MS(name) ((time_ns() - name##_start) / 1e6)

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
    #define TOTAL_TIME 0.001
#endif

#define STEPS ((int)(TOTAL_TIME / DT))  // Number of time steps
#define WRITE_FREQUENCY STEPS           // Writing output frequency
#define ENABLE_OUTPUT 0                 // Set to 1 to enable output, 0 to disable

#define DX_INVERSE (1.0 / DX)
#define DY_INVERSE (1.0 / DY)
#define DZ_INVERSE (1.0 / DZ)

#define DX_INVERSE_SQUARE (DX_INVERSE * DX_INVERSE)
#define DY_INVERSE_SQUARE (DY_INVERSE * DY_INVERSE)
#define DZ_INVERSE_SQUARE (DZ_INVERSE * DZ_INVERSE)

// Simulation parameters
#define MAX_ITERATIONS 1000
#define TOLERANCE 1e-6

#endif // CONSTANTS_H