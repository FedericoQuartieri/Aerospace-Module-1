#ifndef PRESSURE_H
#define PRESSURE_H

#include "solver.h"

/* Assemble -div(u) / DT at pressure points. Assuming div(u)=0 on boundary */
void pressure_step(SolverMemState *solver_mem_state,
                   ScalarField *pressure_buffer,
                   SolverStats *solver_stats);

#endif
