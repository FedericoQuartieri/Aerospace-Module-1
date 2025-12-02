#include "alloc.h"
#include "boundary.h"
#include "solver.h"

#define DEPTH 128
#define HEIGHT 128
#define WIDTH 128

#define NUM_TIMESTEP 10

DEFINE_NU(1.00)
DEFINE_DT(0.01)
DEFINE_DX(0.01)

//DEFINE_FORCE_FIELD()

DEFINE_CONSTANT_BC_U(0, 0, 0, BC_LEFT)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_RIGHT)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_TOP)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_BOTTOM)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_FRONT)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_BACK)

int main(void)
{
    ArenaAllocator arena;
    arena_init(&arena, DEPTH * HEIGTH * WIDTH * sizeof(ftype) * 30);

    Solver *solver = solver_alloc(&arena, DEPTH, HEIGHT, WIDTH);
    solver_init(solver);

    for (int t = 0; t < NUM_TIMESTEPS; ++t) {
        solver_step(solver, t);
    }

    arena_destroy(&arena);

    return 0;
}
