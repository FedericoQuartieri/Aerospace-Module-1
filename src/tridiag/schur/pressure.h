#ifndef PRESSURE_H
#define PRESSURE_H

#include "solver.h"
#include "schur.h"


/*
 * Prepares the three matrices of the pressure cascade, one per axis. They must
 * be built once before the time loop and freed at the end.
 */
void pressure_plans_init(const Decomp *d, SchurPlan plan[3]);
void pressure_plans_free(SchurPlan plan[3]);

/* pressure_step is declared in backend.h: it is the shared interface. */

#endif
