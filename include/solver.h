#ifndef SOLVER_H
#define SOLVER_H

#include "ftype.h"
#include "alloc.h"

struct Solver;

typedef struct Solver Solver;

Solver *solver_alloc(uint32_t domain_depth,
                     uint32_t domain_height,
                     uint32_t domain_width,
                     ArenaAllocator *arena);

void solver_init(Solver *solver);

void solver_set_porosity(Solver *solver, const ftype *src);

void solver_step(Solver *solver, uint64_t timestep);

#endif
