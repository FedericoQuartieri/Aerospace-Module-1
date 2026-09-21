#ifndef PRESSURE_COMMON_H
#define PRESSURE_COMMON_H

#include "decomp.h"
#include "solver.h"
#include "types.h"

/*
 * The part of the pressure cascade that does not depend on the backend.
 *
 * Between the right-hand side and the final update there are three tridiagonal
 * solves, and those are the only thing the two backends do differently. The
 * rest -- building the right-hand side, writing the matrix, recomposing the
 * pressure -- is the same physics, and it is written just once.
 */

/*
 * Assembles -div(u) / DT at the pressure points. The three global lower faces
 * (i == 0, j == 0 or k == 0) are set to zero.
 */
void compute_div(const Decomp *restrict d,
                 Real *restrict u_div,
                 const VectorField *restrict u);

/*
 * The matrix of a line along `axis`, in three arrays of d->n[axis] elements.
 *
 * Unlike momentum, here the matrix is the same for all the lines of the axis
 * and never changes in time: each row depends only on the global position of
 * its point. That is why the shared piece is the line and not the cell --
 * asking for it cell by cell would mean recomputing for every point something
 * known from the start.
 */
void pressure_matrix(const Decomp *restrict d, int axis,
                     Real *restrict a, Real *restrict b, Real *restrict c);

/* p^{n+1} = p^n + phi, and the extrapolated pressure for the next step. */
void update_pressure(const Decomp *restrict d,
                     SolverMemState *restrict solver_mem_state);

#endif
