#ifndef BACKEND_SCHUR_H
#define BACKEND_SCHUR_H

#include <stddef.h>

#include "schur.h"
#include "types.h"

/*
 * Lo stato privato del backend Schur, quello che sta dietro il puntatore
 * opaco SolverMemState.backend.  Non esce da src/tridiag/schur/.
 */
typedef struct SchurBackend {
    size_t scratch_size;   /* elementi di rhs e tmp, non byte             */
    Real *rhs;             /* buffer dei kernel, uno per thread in fila   */
    Real *tmp;
    SchurPlan pressure_plan[3];  /* le tre matrici gia' fattorizzate      */
} SchurBackend;

#endif
