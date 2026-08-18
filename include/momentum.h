#ifndef MOMENTUM_H
#define MOMENTUM_H
#include "solver.h"
#include "simd_real.h"
#include "types.h"

void momentum_step(SolverMemState *solver_mem_state,
                   Data *data,
                   int t_step,
                   SolverStats *solver_stats);

#endif
