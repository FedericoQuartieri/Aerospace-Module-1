#ifndef SOLVER_H
#define SOLVER_H

#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "types.h"
#include "physics.h"
#include "utils.h"

/*
 * Compile-time configuration.  Every value can be overridden with -D, e.g.
 *
 *   cc ... -DWIDTH=32 -DHEIGHT=32 -DDEPTH=32 -DT=1.0 -DSTEPS=100
 */
#ifndef WIDTH
#define WIDTH 256 
#endif

#ifndef HEIGHT
#define HEIGHT 256
#endif

#ifndef DEPTH
#define DEPTH 256
#endif

#define GRID_CELLS ((size_t)WIDTH * (size_t)HEIGHT * (size_t)DEPTH)

// Physical domain
#ifndef LX
#define LX M_PI
#endif

#ifndef LY
#define LY M_PI
#endif

#ifndef LZ
#define LZ M_PI
#endif

#define DX ((2 * LX) / (Real)(2*WIDTH - 1))  // Grid spacing in x
#define DY ((2 * LY) / (Real)(2*HEIGHT - 1)) // Grid spacing in y
#define DZ ((2 * LZ) / (Real)(2*DEPTH - 1))  // Grid spacing in z

#define DX_INVERSE (1.0 / DX)
#define DY_INVERSE (1.0 / DY)
#define DZ_INVERSE (1.0 / DZ)
#define DX_INVERSE_SQUARE (DX_INVERSE * DX_INVERSE)
#define DY_INVERSE_SQUARE (DY_INVERSE * DY_INVERSE)
#define DZ_INVERSE_SQUARE (DZ_INVERSE * DZ_INVERSE)

#ifndef T
#define T 1e-0
#endif

#ifndef STEPS
#define STEPS 200
#endif

#define DT ((Real)(T) / (Real)(STEPS))
#define WR_FREQ 5
// Kinematic viscosity
#define NU 1.0


typedef struct SolverMemState {
    VectorField eta;
    VectorField zeta;
    VectorField u;
    VectorField k;
    ScalarField pressure;
    ScalarField pressure_star;
} SolverMemState;

extern const Data paper_data;

void solver_init(SolverMemState *solver_mem_state,
                 Data *data,
                 const char *data_name);

void solver_solve(SolverMemState *solver_mem_state,
                  Data *data,
                  SolverStats *solver_stats,
                  int write_enabled);

#endif
