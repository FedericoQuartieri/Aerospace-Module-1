#ifndef PHYSICS_H
#define PHYSICS_H
#include "types.h"

struct SolverMemState;

Real beta_from_k(Real k);
Real gamma_from_k(Real k);

Real time_physical_coord(int t_step);
Real centered_physical_coord(int index, int component);
Real staggered_physical_coord(int index, int component);

Real bc_left(VectorFunction bc_velocity,
             int i, int j, int k, int t_step, int component);
Real bc_right(VectorFunction bc_velocity,
              int i, int j, int k, int t_step, int component);

Real g_value(int i, int j, int k, int t_step, Real k_i,
             const struct SolverMemState *solver_mem_state,
             const Data *data, int component);

#endif
