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
    
    solver_init(&solver_mem_state, &data, data_name);
    
    solver_solve(&solver_mem_state, &data);

    return 0;
}
