#ifndef BACKEND_SCHUR_H
#define BACKEND_SCHUR_H

#include <stddef.h>

#include "schur.h"
#include "types.h"

/*
 * The private state of the Schur backend, what sits behind the opaque pointer
 * SolverMemState.backend. It does not leave src/tridiag/schur/.
 */
typedef struct SchurBackend {
    size_t scratch_size;   /* elements of rhs and tmp, not bytes */
    Real *rhs;             /* kernel buffers, one per thread in a row */
    Real *tmp;
    SchurPlan pressure_plan[3];  /* the three already-factorised matrices */
} SchurBackend;

#endif
