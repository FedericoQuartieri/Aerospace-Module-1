#ifndef PRESSURE_H
#define PRESSURE_H

#include "solver.h"

void pressure_step(SolverMemState *solver_mem_state,
                   Real *restrict rhs,
                   Real *restrict tmp,
                   Data *data,
                   int t_step,
                   SolverStats *solver_stats);

#endif
