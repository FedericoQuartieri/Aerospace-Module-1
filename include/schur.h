#ifndef SCHUR_H
#define SCHUR_H

#include "decomp.h"
#include "types.h"

/*
 * Tridiagonal solvers.
 *
 * The system is  a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = f[i],  i = 0..n-1,
 * where a[0] and c[n-1] are never read.
 *
 * thomas_solve is the ordinary sequential algorithm.  schur_solve produces
 * the same answer, but splits the unknowns into contiguous blocks and only
 * ever couples a block to its neighbours through the single point they share.
 * That is the shape the solver needs once the blocks live on different
 * processes: everything each block does is local except one small system.
 *
 * Both need scratch space of n elements; schur_solve allocates its own.
 */
void thomas_solve(int n,
                  const Real *a, const Real *b, const Real *c,
                  const Real *f, Real *x, Real *scratch);

/*
 * Solve by Schur complement over `blocks` contiguous blocks.
 * Every block must hold at least two points.
 */
void schur_solve(int n, int blocks,
                 const Real *a, const Real *b, const Real *c,
                 const Real *f, Real *x);

/*
 * The same method, with one block per process along `axis` of the process
 * grid.  Every array holds only this process's n_local points, and the answer
 * matches what thomas_solve would produce for the whole line.
 *
 * With a single process along the axis it falls back to thomas_solve, which
 * is what makes a one-process run reproduce the serial result exactly.
 */
void schur_solve_mpi(int axis, int n_local,
                     const Real *a, const Real *b, const Real *c,
                     const Real *f, Real *x);

#endif
