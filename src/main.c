#include <stdio.h>
#include "solver.h"

int main(int argc, char **argv) {
    // Data of bc_velocity and forcing used by the solver
    char *data_name = NULL;
    if(argc == 2) {
        data_name = argv[1];  
    }

    SolverMemState solver_mem_state;
    Data data = paper_data;
    SolverStats solver_stats = {0};
    Decomp decomp;

    decomp_init_serial(&decomp);
    solver_init(&decomp, &solver_mem_state, &data, data_name);
    
    int write_enabled = 0;
    solver_solve(&decomp, &solver_mem_state, &data, &solver_stats,
                 write_enabled);

    return 0;
}
