#include "solver.h"
#include "momentum.h"
#include "pressure.h"
#include "ftype.h"
#include "field.h"
#include "consts.h"

struct Solver {
    ArenaAllocator *arena;
    field_size domain_size;
    field porosity;
    field gamma;
    field pressure;
    field pressure_delta;
    field3 velocity_Dxx;
    field3 velocity_Dyy;
    field3 velocity_Dzz;
};

Solver *solver_alloc(uint32_t domain_depth,
                     uint32_t domain_height,
                     uint32_t domain_width,
                     ArenaAllocator *arena)
{
    Solver *solver = arena_push(arena, sizeof(Solver));
    memset(solver, 0, sizeof(Solver));

    solver->arena = arena;

    field_size domain_size;
    domain_size.depth = domain_depth;
    domain_size.height = domain_height;
    domain_size.width = domain_width;
    solver->domain_size = domain_size;

    solver->porosity = field_alloc(domain_size, arena);
    solver->gamma = field_alloc(domain_size, arena);
    solver->pressure = field_alloc(domain_size, arena);
    solver->pressure_delta = field_alloc(domain_size, arena);

     /* Extra faces for in-bound finite difference. */
    domain_size.depth = domain_depth + 2;
    solver->velocity_Dxx = field3_alloc(domain_size, arena);
    solver->velocity_Dyy = field3_alloc(domain_size, arena);
    solver->velocity_Dzz = field3_alloc(domain_size, arena);

    return solver;
}

void solver_init(Solver *solver)
{
    field_size domain_size = solver->domain_size;

    momentum_init(domain_size, solver->velocity_Dxx);
    momentum_init(domain_size, solver->velocity_Dyy);
    momentum_init(domain_size, solver->velocity_Dzz);

    pressure_init(domain_size, solver->pressure);
    pressure_init(domain_size, solver->pressure_delta);

    arena_enter(solver->arena);

    /* Setting constant unit porosity. */
    field tmp = field_alloc(domain_size, solver->arena);
    field_fill(domain_size, 1.0, tmp);
    solver_set_porosity(solver, tmp);

    arena_exit(solver->arena);
}

void solver_set_porosity(Solver *solver, const ftype *src)
{
    field_copy(solver->domain_size, src, solver->gamma);
    compute_gamma(src, solver->domain_size, solver->porosity);
}

void solver_step(Solver *solver, uint64_t timestep)
{
    momentum_solve(solver->porosity,
                   solver->gamma,
                   solver->pressure,
                   solver->pressure_delta,
                   solver->domain_size,
                   solver->velocity_Dxx,
                   solver->velocity_Dyy,
                   solver->velocity_Dzz,
                   solver->arena);

    pressure_solve(to_const_field3(solver->velocity_Dzz),
                   solver->domain_size,
                   solver->pressure,
                   solver->pressure_delta,
                   solver->arena);
}
