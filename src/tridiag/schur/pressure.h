#ifndef PRESSURE_H
#define PRESSURE_H

#include "solver.h"
#include "schur.h"


/*
 * Prepara le tre matrici della cascata di pressione, una per asse.  Vanno
 * costruite una volta prima del ciclo temporale e liberate alla fine.
 */
void pressure_plans_init(const Decomp *d, SchurPlan plan[3]);
void pressure_plans_free(SchurPlan plan[3]);

/* pressure_step e' dichiarata in backend.h: e' l'interfaccia condivisa. */

#endif
