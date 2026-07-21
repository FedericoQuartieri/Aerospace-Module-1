#include "solver.h"
#include "field.h"



void solver_init(SolverMemState *solver_mem_state,
                 Data *data,
                 const char *data_name) {
   
    //TODO: Parse data_name and assign the correspondent Data structure,
    // return error if not matched
    

    // Allocate memory
    scalarField_alloc(&solver_mem_state->pressure);
    scalarField_alloc(&solver_mem_state->pressure_star);
    //vectorField_alloc(&solver_mem_state->k);
    vectorField_alloc(&solver_mem_state->eta);
    vectorField_alloc(&solver_mem_state->zeta);
    vectorField_alloc(&solver_mem_state->u);
    
    // Initialize velocity and pressure with value at t=0, t=1/2
    vectorField_fill(&solver_mem_state->eta, data->velocity_fn);
    vectorField_fill(&solver_mem_state->zeta, data->velocity_fn);
    vectorField_fill(&solver_mem_state->u, data->velocity_fn);

    scalarField_fill(&solver_mem_state->pressure, data->pressure_fn);
    scalarField_fill(&solver_mem_state->pressure_star, data->pressure_fn);
    





}
