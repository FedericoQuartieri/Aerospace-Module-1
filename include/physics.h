#ifndef PHYSICS_H
#define PHYSICS_H
#include "decomp.h"
#include "types.h"

struct SolverMemState;

Real beta_from_k(Real k);
Real gamma_from_k(Real k);

Real time_physical_coord(Real t_step);
Real centered_physical_coord(int index, int component);
Real staggered_physical_coord(int index, int component);

/*
 * The boundary helpers describe positions in physical space only, so their
 * i, j, k arguments are *global* indices and the caller converts.  That also
 * turns their internal face tests (i == 0, i == WIDTH - 1, ...) into tests on
 * the global boundary, which is what they must be once the grid is split.
 */
Real bc_left(VectorFunction bc_velocity,
             int i, int j, int k, int t_step, int component);
Real bc_right(VectorFunction bc_velocity,
              int i, int j, int k, int t_step, int component);

/*
 * g_value addresses memory as well, so it takes local indices plus the
 * decomposition and derives the global ones itself.
 */
Real g_value(const Decomp *d,
             int i, int j, int k, int t_step, Real k_i,
             const struct SolverMemState *solver_mem_state,
             const Data *data, int component);

#endif
