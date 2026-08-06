#ifndef PRESSURE_H
#define PRESSURE_H

#include "solver.h"

/*
 * Assemble -div(u) / DT at pressure points.  The three lower boundary
 * faces (i == 0, j == 0 or k == 0) are set to zero.
 */
void compute_div(const Decomp *restrict d,
                 Real *restrict u_div,
                 const VectorField *restrict u);

void pressure_step(const Decomp *decomp,
                   SolverMemState *solver_mem_state,
                   ScalarField *pressure_buffer,
                   Real *restrict rhs,
                   Real *restrict tmp,
                   SolverStats *solver_stats);

#endif
