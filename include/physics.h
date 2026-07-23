#ifndef PHYSICS_H
#define PHYSICS_H
#include "solver.h"

Real beta_from_k(Real k);
Real gamma_from_k(Real k);

Real time_physical_coord(int t_step);
Real centered_physical_coord(int index, int component);
Real staggered_physical_coord(int index, int component);

Real bc_left(VectorFunction bc_velocity,
             int i, int j, int k, int t_step, int component);
Real bc_right(VectorFunction bc_velocity,
              int i, int j, int k, int t_step, int component);

#endif
