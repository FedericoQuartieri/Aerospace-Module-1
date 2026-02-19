#include <stdio.h>

#include "alloc.h"
#include "boundary.h"
#include "consts.h"
#include "solver.h"
#include "timeit.h"

#include "field.h"
#include "output.h"

#define DEPTH 64
#define HEIGHT 128
#define WIDTH 64

#define NUM_TIMESTEPS 100

DEFINE_NU(1.0)
DEFINE_DT(0.005)
DEFINE_DX(M_PI / WIDTH)

DEFINE_CONSTANT_FORCING(0, 0, 0)

DEFINE_CONSTANT_BC_U(0, 0, 0, BC_LEFT)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_RIGHT)
DEFINE_CONSTANT_BC_U(0, 0.1, 0, BC_TOP)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_BOTTOM)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_FRONT)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_BACK)

int main(void)
{
    ArenaAllocator arena;
    arena_init(&arena, 1lu << 34);

    Solver *solver = solver_alloc(DEPTH, HEIGHT, WIDTH, &arena);
    solver_init(solver);

    field_size size = { WIDTH, HEIGHT, DEPTH };
    OutputVTK *output = output_vtk_create(size, _DX, &arena);

    output_vtk_attach_field(output, solver_get_porosity(solver),
                            "porosity", &arena);
    output_vtk_attach_field(output, solver_get_pressure(solver),
                            "pressure", &arena);
    output_vtk_attach_field3(output, solver_get_velocity(solver),
                             "velocity", &arena);
    output_vtk_write(output, "solution-cavity-0.vtk");

    for (int t = 1; t < NUM_TIMESTEPS + 1; ++t) {
        TIMEITN(solver_step(solver, t), 1);

        char output_file_name[32];
        sprintf(output_file_name, "solution-cavity-%d.vtk", t);
        TIMEITN(output_vtk_write(output, output_file_name), 1);

        printf("\n");
    }

    arena_destroy(&arena);

    return 0;
}
