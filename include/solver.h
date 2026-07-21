#ifndef SOLVER_H
#define SOLVER_H

#include <stddef.h>

#include "types.h"

#define WIDTH 64
#define HEIGHT 64
#define DEPTH 64

#define GRID_CELLS ((size_t)WIDTH * (size_t)HEIGHT * (size_t)DEPTH)

typedef struct SolverMemState {
    VectorField eta;
    VectorField zeta;
    VectorField u;
    ScalarField pressure;
    ScalarField pressure_star;
} SolverMemState;

extern const Data paper_data;

void solver_init(SolverMemState *solver_mem_state,
                 Data *data,
                 const char *data_name);

#endif
