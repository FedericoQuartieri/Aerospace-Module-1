#include <string.h>

#include "ftype.h"
#include "field.h"
#include "solver.h"

static void compute_gamma(const_field porosity, field_size size, field dst)
{
    uint64_t num_points = size.depth * size.height * size.width;
    for (uint64_t i = 0; i < num_points; i += VLEN) {
        vftype k = vload(porosity + i);
        /* w = (k dt nu) / (2k + dt nu) / (dx dx) */
        vstore(dst + i,
               (k * _DT * _NU) / (2 * k + _DT * _NU) / (_DX * _DX));
    }
}

struct Solver {
    ArenaAllocator *arena;
    ffield_size domain_size;
    ffield porosity;
    ffield gamma;
    ffield pressure;
    ffield pressure_delta;
    ffield3 velocity_Dxx;
    ffield3 velocity_Dyy;
    ffield3 velocity_Dzz;
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

    solver->porosity = ffield_alloc(domain_size, arena);
    solver->gamma = ffield_alloc(domain_size, arena);
    solver->pressure = ffield_alloc(domain_size, arena);
    solver->pressure_delta = ffield_alloc(domain_size, arena);

    solver->velocity_Dxx = ffield3_alloc(domain_size, arena);
    solver->velocity_Dyy = ffield3_alloc(domain_size, arena);
    solver->velocity_Dzz = ffield3_alloc(domain_size, arena);

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
    uint64_t num_points = field_num_points(domain_size);
    ftype *tmp = arena_push_count(solver->arena, ftype, num_points);
    fmemset(tmp, 1.0, num_points);
    solver_set_porosity(solver, tmp);

    arena_exit(solver->arena);
}

void solver_set_porosity(Solver *solver, const ftype *src)
{
    uint64_t num_points = field_num_points(solver->domain_size);
    memcpy(solver->gamma, src, num_points * sizeof(ftype));
    compute_gamma(src, domain_size, solver->gamma);
}

void solver_step(Solver *solver, uint64_t timestep)
{
    return;
}
