#include "solver.h"
#include "field.h"
#include "momentum.h"
#include "pressure.h"
#include "output.h"

static size_t pipeline_axis_capacity(const Domain *domain, int axis,
                                     int batch_lines) {
    size_t line_count;
    size_t batch_count;

    if (axis == AXIS_X) {
        line_count = (size_t)domain->local[AXIS_Y] *
                     (size_t)domain->local[AXIS_Z];
    } else if (axis == AXIS_Y) {
        line_count = (size_t)domain->local[AXIS_X] *
                     (size_t)domain->local[AXIS_Z];
    } else {
        line_count = (size_t)domain->local[AXIS_X] *
                     (size_t)domain->local[AXIS_Y];
    }

    batch_count = (line_count + (size_t)batch_lines - 1) /
                  (size_t)batch_lines;
    return batch_count * (size_t)batch_lines *
           (size_t)domain->local[axis];
}

static void pipeline_alloc(PipelineWorkspace *pipeline,
                           const Domain *domain) {
    size_t capacity = 0;

    pipeline->batch_lines = PIPELINE_BATCH_LINES;
    if (pipeline->batch_lines < 1) {
        pipeline->batch_lines = 1;
    }
    for (int axis = 0; axis < AXIS_COUNT; axis++) {
        size_t axis_capacity =
            pipeline_axis_capacity(domain, axis, pipeline->batch_lines);
        if (axis_capacity > capacity) {
            capacity = axis_capacity;
        }
    }

    pipeline->component_capacity = capacity;
    pipeline->c_prime = xmalloc(3 * capacity * sizeof(Real));
    pipeline->d_prime = xmalloc(3 * capacity * sizeof(Real));
    pipeline->forward = xmalloc(2 * (size_t)pipeline->batch_lines *
                                sizeof(Real));
    pipeline->backward = xmalloc((size_t)pipeline->batch_lines *
                                 sizeof(Real));
}

void solver_init(SolverMemState *solver_mem_state,
                 Data *data,
                 const char *data_name) {
    (void)data_name;
   
    // TODO: Parse data_name and assign the correspondent Data structure,
    // return error if not matched
    
    const int global[AXIS_COUNT] = {WIDTH, HEIGHT, DEPTH};

    domain_init(&solver_mem_state->domain, global);

    scalarField_alloc(&solver_mem_state->pressure, &solver_mem_state->domain);
    scalarField_alloc(&solver_mem_state->pressure_star,
                      &solver_mem_state->domain);
    vectorField_alloc(&solver_mem_state->k, &solver_mem_state->domain);
    vectorField_alloc(&solver_mem_state->eta, &solver_mem_state->domain);
    vectorField_alloc(&solver_mem_state->zeta, &solver_mem_state->domain);
    vectorField_alloc(&solver_mem_state->u, &solver_mem_state->domain);
    pipeline_alloc(&solver_mem_state->pipeline, &solver_mem_state->domain);
    
    // Fill velocity with values at t=0
    vectorField_fill(&solver_mem_state->eta, &solver_mem_state->domain,
                     data->velocity_fn, 0);
    vectorField_fill(&solver_mem_state->zeta, &solver_mem_state->domain,
                     data->velocity_fn, 0);
    vectorField_fill(&solver_mem_state->u, &solver_mem_state->domain,
                     data->velocity_fn, 0);
    
    // Fill pressure with values at t=0
    scalarField_fill(&solver_mem_state->pressure, &solver_mem_state->domain,
                     data->pressure_fn, 0);
    scalarField_fill(&solver_mem_state->pressure_star,
                     &solver_mem_state->domain, data->pressure_fn, 0);
    
    // The porosity is static and is initialized only once. 
    // (In this version is not time-dependent)
    vectorField_fill(&solver_mem_state->k, &solver_mem_state->domain,
                     data->porosity_fn, 0);
}

void solver_solve(SolverMemState *solver_mem_state, Data *data, SolverStats *solver_stats,
                  int write_enabled) {
    ScalarField pressure_buffer;
    scalarField_alloc(&pressure_buffer, &solver_mem_state->domain);

    // If write_enabled is set, write the initial state at t=0
    if (write_enabled) write_to_file(solver_mem_state, data->name, 0);

    MPI_Barrier(solver_mem_state->domain.cart);
    uint64_t start_ns = time_ns();
    for (int t_step = 1; t_step <= STEPS; t_step++) {
        // Momentum system
        momentum_step(solver_mem_state, data, t_step, solver_stats);


        // Pressure system
        pressure_step(solver_mem_state, &pressure_buffer, solver_stats);

        // Write to file
        if (write_enabled) {
            if (t_step % WR_FREQ == 0) {
                uint64_t wr_start = time_ns();
                write_to_file(solver_mem_state, data->name, t_step);
                solver_stats->wr_output += time_ns() - wr_start;
            }
        }
    }
    solver_stats->solve_steps = (time_ns() - start_ns) - solver_stats->wr_output;

    SolverStats *rank_stats = solver_mem_state->domain.rank == 0
        ? xmalloc((size_t)solver_mem_state->domain.size * sizeof(SolverStats))
        : NULL;
    MPI_Gather(solver_stats,
               (int)(sizeof(SolverStats) / sizeof(uint64_t)), MPI_UINT64_T,
               rank_stats,
               (int)(sizeof(SolverStats) / sizeof(uint64_t)), MPI_UINT64_T,
               0, solver_mem_state->domain.cart);
    if (solver_mem_state->domain.rank == 0) {
        const SolverStats *critical_stats = &rank_stats[0];
        for (int rank = 1; rank < solver_mem_state->domain.size; rank++) {
            if (rank_stats[rank].solve_steps > critical_stats->solve_steps) {
                critical_stats = &rank_stats[rank];
            }
        }
        printf("MPI process grid: %d x %d x %d\n",
               solver_mem_state->domain.dims[0],
               solver_mem_state->domain.dims[1],
               solver_mem_state->domain.dims[2]);
        printf("Pipeline batch: %d lines\n",
               solver_mem_state->pipeline.batch_lines);
        print_stats(critical_stats, (size_t)STEPS);
        free(rank_stats);
    }

    free(pressure_buffer.v);
}

void solver_destroy(SolverMemState *solver_mem_state) {
    vectorField_free(&solver_mem_state->eta);
    vectorField_free(&solver_mem_state->zeta);
    vectorField_free(&solver_mem_state->u);
    vectorField_free(&solver_mem_state->k);
    free(solver_mem_state->pressure.v);
    free(solver_mem_state->pressure_star.v);
    free(solver_mem_state->pipeline.c_prime);
    free(solver_mem_state->pipeline.d_prime);
    free(solver_mem_state->pipeline.forward);
    free(solver_mem_state->pipeline.backward);
    domain_destroy(&solver_mem_state->domain);
}
