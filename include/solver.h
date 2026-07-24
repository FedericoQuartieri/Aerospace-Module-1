#ifndef SOLVER_H
#define SOLVER_H

#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "types.h"
#include "physics.h"
#include "utils.h"

#define WIDTH 128
#define HEIGHT 128
#define DEPTH 128

#define GRID_CELLS ((size_t)WIDTH * (size_t)HEIGHT * (size_t)DEPTH)

// Physical domain
#define LX M_PI
#define LY M_PI 
#define LZ M_PI

#define DX ((2 * LX) / (Real)(2*WIDTH - 1))  // Grid spacing in x
#define DY ((2 * LY) / (Real)(2*HEIGHT - 1)) // Grid spacing in y
#define DZ ((2 * LZ) / (Real)(2*DEPTH - 1))  // Grid spacing in z

#define DX_INVERSE (1.0 / DX)
#define DY_INVERSE (1.0 / DY)
#define DZ_INVERSE (1.0 / DZ)
#define DX_INVERSE_SQUARE (DX_INVERSE * DX_INVERSE)
#define DY_INVERSE_SQUARE (DY_INVERSE * DY_INVERSE)
#define DZ_INVERSE_SQUARE (DZ_INVERSE * DZ_INVERSE)

#define T 1e-2
#define DT 1e-3
#define STEPS (int)(T/DT)
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
                  SolverStats *solver_stats);

#endif
