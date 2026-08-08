#include <stdio.h>
#include "solver.h"
#include "parallel.h"

int main(int argc, char **argv) {
    par_init(&argc, &argv);

    // Data of bc_velocity and forcing used by the solver
    char *data_name = NULL;
    if(argc == 2) {
        data_name = argv[1];  
    }

    SolverMemState solver_mem_state;
    Data data = paper_data;
    SolverStats solver_stats = {0};
    Decomp decomp;

    /* {1, 1, 0}: i processi si dispongono a fette lungo Z. */
    const int process_grid[3] = {1, 1, 0};
    par_topology_init(process_grid);
    decomp_init_mpi(&decomp);
    solver_init(&decomp, &solver_mem_state, &data, data_name);
    
    int write_enabled = 0;
    solver_solve(&decomp, &solver_mem_state, &data, &solver_stats,
                 write_enabled);

    par_finalize();
    return 0;
}
