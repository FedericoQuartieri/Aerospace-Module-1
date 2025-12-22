#include "alloc.h"
#include "boundary.h"
#include "consts.h"
#include "solver.h"
#include "timeit.h"

#define DEPTH 256
#define HEIGHT 256
#define WIDTH 256

#define NUM_TIMESTEPS 5

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
    arena_init(&arena, DEPTH * HEIGHT * WIDTH * sizeof(ftype) * 30);

    Solver *solver = solver_alloc(DEPTH, HEIGHT, WIDTH, &arena);
    solver_init(solver);

    for (int t = 1; t < NUM_TIMESTEPS + 1; ++t) {
        TIMEIT(solver_step(solver, t));
    }

    printf("%f %f\n", solver_get_velocity(solver).x[0],
                      solver_get_pressure(solver)[0]);

    arena_destroy(&arena);

    return 0;
}
