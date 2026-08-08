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
 * grid, applied to `lines` independent systems at once.  Every array holds
 * only this process's points, with system l at offset l * n_local, and the
 * answer matches what thomas_solve would produce for the whole line.
 *
 * The lines are solved together so that the whole group costs one exchange
 * and one collective rather than one per line.
 *
 * With a single process along the axis it falls back to thomas_solve, which
 * is what makes a one-process run reproduce the serial result exactly.
 */
void schur_solve_mpi(int axis, int lines, int n_local,
                     const Real *a, const Real *b, const Real *c,
                     const Real *f, Real *x);

/*
 * The same method, with the preprocessing separated from the runtime
 * (Lecture 5, p. 32).
 *
 * schur_solve_mpi rebuilds everything at every call, which is the only option
 * when the matrix changes: the momentum systems carry gamma, and gamma follows
 * the permeability from one cell to the next.  The pressure cascade is the
 * opposite case.  Its matrix is the same for every line and never changes, so
 * the two influence functions and the whole interface system depend on nothing
 * that a time step brings: they are computed once, and only the right-hand
 * side is left to schur_plan_solve.
 *
 * That turns a call from three local Thomas solves per line into one, and the
 * exchanged data from four values per line into one.
 */
typedef struct SchurPlan {
    int axis;
    int n;              /* points of this block along axis, interface included */
    int len;            /* of those, the internal ones                        */
    int blocks;         /* processes along axis                               */
    int p;              /* my position among them                            */
    int has_right;      /* whether my last point is an interface             */
    Real *a, *b, *c;    /* the block's matrix, one line                      */
    Real *lft, *rgt;    /* answers to a unit value on either interface       */
    Real *ra, *rb, *rc; /* the interface system, already assembled           */
} SchurPlan;

/* Preprocessing: a, b, c describe one line of n points of this block. */
void schur_plan_init(SchurPlan *plan, int axis, int n,
                     const Real *a, const Real *b, const Real *c);

/* Runtime: `lines` right-hand sides, system l at offset l * n. */
void schur_plan_solve(const SchurPlan *plan, int lines,
                      const Real *f, Real *x);

void schur_plan_free(SchurPlan *plan);

#endif
