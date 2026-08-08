#ifndef PRESSURE_H
#define PRESSURE_H

#include "solver.h"
#include "schur.h"

/*
 * Assemble -div(u) / DT at pressure points.  The three lower boundary
 * faces (i == 0, j == 0 or k == 0) are set to zero.
 */
void compute_div(const Decomp *restrict d,
                 Real *restrict u_div,
                 const VectorField *restrict u);

/*
 * Prepara le tre matrici della cascata di pressione, una per asse.  Vanno
 * costruite una volta prima del ciclo temporale e liberate alla fine.
 */
void pressure_plans_init(const Decomp *d, SchurPlan plan[3]);
void pressure_plans_free(SchurPlan plan[3]);

void pressure_step(const Decomp *decomp,
                   SolverMemState *solver_mem_state,
                   const SchurPlan plan[3],
                   ScalarField *pressure_buffer,
                   Real *restrict rhs,
                   Real *restrict tmp,
                   SolverStats *solver_stats);

#endif
