#ifndef BACKEND_H
#define BACKEND_H

#include "decomp.h"
#include "solver.h"
#include "types.h"

/*
 * What the solver asks of the tridiagonal backend, and nothing else.
 *
 * Below this line there are two implementations, chosen at compile time with
 * TRIDIAG=schur|pipeline, and each is master of its own loop, its own scratch
 * and its own order of the components. Above, solver.c does not know which of
 * the two it has in front of it.
 *
 * The seam is here and not lower down -- not at "solve this block of lines" --
 * for a precise reason. A signature of the kind
 *
 *     solve(axis, lines, n_local, a, b, c, f, x)
 *
 * does not only impose *what* to compute: it also imposes where to put it, how
 * many lines at a time to look at it, and when to have finished. Those are the
 * three things that the pipelined Thomas does differently:
 *
 *   - it never materialises the four diagonals, it consumes them on the spot;
 *   - it wants all the lines of the axis together, not one plane at a time, or
 *     the pipeline is all filling and draining;
 *   - it keeps the three components in flight (forward x,y,z then backward
 *     z,y,x) so as not to drain between the forward and the backward sweep.
 *
 * By raising the seam to the directional step, those three choices stay inside
 * the backend, which is the only place where they make sense. What the two
 * share is the physics of a point, and it lives in momentum_row.h.
 */

/*
 * The backend scratch: whoever needs it allocates it here and keeps it in
 * SolverMemState.backend, which for the shared code is an opaque pointer.
 *
 * The Schur complement puts the SIMD kernel buffers and the three
 * already-factorised pressure matrices in it; the pipelined Thomas puts its c'
 * and d' in it. Neither type reaches solver.c, which before instead had to
 * know SchurPlan in order to declare it on the stack.
 */
void backend_init(const Decomp *d, SolverMemState *solver_mem_state);
void backend_free(SolverMemState *solver_mem_state);

/* Name of the compiled backend, for the statistics and for the tests. */
const char *backend_name(void);
/*
 * Lines per batch chosen by backend_init, the same on all processes. The
 * pipeline uses them; Schur answers 0, because the parameter does not concern
 * it.
 */
int backend_batch_lines(void);

/*
 * The three momentum systems of a time step, all three axes and all three
 * components.
 */
void momentum_step(const Decomp *d, SolverMemState *solver_mem_state,
                   Data *data, int t_step, SolverStats *solver_stats);

/*
 * The pressure cascade of a time step, plus the update. `pressure_buffer` is
 * work space as large as a scalar field, which the caller owns.
 */
void pressure_step(const Decomp *d, SolverMemState *solver_mem_state,
                   ScalarField *pressure_buffer, SolverStats *solver_stats);

#endif
