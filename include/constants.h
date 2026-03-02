#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <math.h>

// Grid dimensions
#define WIDTH 64
#define HEIGHT 64
#define DEPTH 64

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

#define DT 0.1                        // Time step
#define TOTAL_TIME 2                    // Total simulation time
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

#endif // CONSTANTS_H