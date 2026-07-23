#ifndef MOMENTUM_H
#define MOMENTUM_H
#include "solver.h"
#include "types.h"
#include <stdbool.h>

void momentum_step(SolverMemState *solver_mem_state, Real *rhs, Real *tmp, 
                Data *data, int t_step);

void update_eta(SolverMemState *solver_mem_state, Real *rhs, Real *tmp,
                Data *data, int t_step, int v_comp);
void update_zeta(SolverMemState *solver_mem_state, Real *rhs, Real *tmp,
                Data *data, int t_step, int v_comp);
void update_u(SolverMemState *solver_mem_state, Real *rhs, Real *tmp,
                Data *data, int t_step, int v_comp);

#endif
