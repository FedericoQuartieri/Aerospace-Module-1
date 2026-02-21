#include <stdio.h>

#include "alloc.h"
#include "boundary.h"
#include "consts.h"
#include "solver.h"
#include "timeit.h"
#include "field.h"
#include "output.h"
#include "thread-array.h"

#define DEPTH 256
#define HEIGHT 256
#define WIDTH 256

#define NUM_TIMESTEPS 25

#define NUM_THREADS 8

DEFINE_NU(1.0)
DEFINE_DT(0.025)
DEFINE_DX(M_PI / WIDTH)

DEFINE_CONSTANT_FORCING(0, 0, 0)

DEFINE_CONSTANT_BC_U(0, 0, 0, BC_LEFT)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_RIGHT)
DEFINE_CONSTANT_BC_U(0, 0.1, 0, BC_TOP)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_BOTTOM)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_FRONT)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_BACK)

typedef struct {
    Solver *solver;
    OutputVTK *output;
} SimulationData;

void *run_simulation(void *t_data)
{
    ArenaAllocator *arena = thread_get_arena(t_data);
    SimulationData *sim_data = thread_get_shared_data(t_data);

    Solver *solver = sim_data->solver;
    OutputVTK *output = sim_data->output;

    uint32_t t_id = ((Thread *) t_data)->t_id;

    if (t_id == 0) {
        output_vtk_write(output, "output/solution-cavity-0.vtk", arena);
    }

    thread_wait_on_barrier(t_data);

    for (int t = 1; t < NUM_TIMESTEPS + 1; ++t) {
        TIMEITN(solver_step(solver, t, t_data), 1);

        if (t_id == 0) {
            char output_file_name[32];
            sprintf(output_file_name, "output/solution-cavity-%d.vtk", t);
            TIMEITN(output_vtk_write(output, output_file_name, arena), 1);
            printf("\n");
        }

        thread_wait_on_barrier(t_data);
    }

    pthread_exit(NULL);
}

int main(void)
{
    ArenaAllocator arena;
    arena_init(&arena, 1lu << 34);

    Solver *solver = solver_alloc(DEPTH, HEIGHT, WIDTH, &arena);
    solver_init(solver, &arena);

    field_size size = { WIDTH, HEIGHT, DEPTH };
    OutputVTK *output = output_vtk_create(size, _DX, &arena);

    output_vtk_attach_field(output, solver_get_porosity(solver),
                            "porosity", &arena);
    output_vtk_attach_field(output, solver_get_pressure(solver),
                            "pressure", &arena);
    output_vtk_attach_field3(output, solver_get_velocity(solver),
                             "velocity", &arena);

    ThreadArray *t_array = thread_array_create(NUM_THREADS, &arena);
    SimulationData data = { solver, output };
    thread_array_set_shared_data(t_array, &data);

    thread_array_run(t_array, run_simulation, &arena);
    thread_array_destroy(t_array);

    arena_destroy(&arena);

    return 0;
}
