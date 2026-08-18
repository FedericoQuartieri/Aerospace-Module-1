#include <stdio.h>
#include "solver.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    // Data of bc_velocity and forcing used by the solver
    char *data_name = NULL;
    if(argc == 2) {
        data_name = argv[1];  
    }

    SolverMemState solver_mem_state;
    Data data = paper_data;
    SolverStats solver_stats = {0};

    solver_init(&solver_mem_state, &data, data_name);
    
    int write_enabled = 0;
    solver_solve(&solver_mem_state, &data, &solver_stats, write_enabled);

    solver_destroy(&solver_mem_state);
    MPI_Finalize();

    return 0;
}
