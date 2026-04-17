#include <stdio.h>
#include <stdlib.h>
#include "../include/constants.h"
#include "velocity_field.h"
#include "pressure.h"
#include "force_field.h"
#include "g_field.h"
#include "utils.h"
#include "momentum_system.h"
#include "pressure_system.h"
#include "solve.h"

#include "data.h"

/* Navier-Stokes-Brinkman equation solver */

int main(){
    
    Pressure pressure;
    initialize_pressure(&pressure);

    VelocityField Eta;
    VelocityField Zeta;
    VelocityField U;

    const Data *data = &PAPER_DATA;

    /*
        Missing: in the first timestep we must enforce the exact solution, 
        that is given by the problem, required to be well posed.
        So we should set Eta, Zeta, U to the exact solution in t=0,
        then solve for t=1. 
        The boundary conditions are computed as the delta between (t=0, t=1)

        (For now in test_manufactured.c is manually set the exact velocity at t=0)
    */
   
    // Initilized 3 velocity field, and for each one set the SAME boundary conditions,
    initialize_velocity_field(&Eta);
    initialize_velocity_field(&Zeta);
    initialize_velocity_field(&U);


    // Set K that is needed to compute Gamma
    // K depends also on the spatial coordinate
    DTYPE *K = (DTYPE *) malloc(GRID_SIZE);
    const_fill(K); // fill with all 1 for now 
    
    DTYPE *Beta = (DTYPE *) malloc(GRID_SIZE);
    DTYPE *Gamma = (DTYPE *) malloc(GRID_SIZE);
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);
                Beta[idx] = 1.0 + (DT * NU) / (2.0 * K[idx]);
                Gamma[idx] = (DT * NU) / (2.0 * Beta[idx]);
            }
        }
    }

    // Inizialize g
    GField g_field;
    initialize_g_field(&g_field);    

    solve(g_field, data, pressure, K, Eta, Zeta, U, Beta, Gamma,
        WRITE_FREQUENCY, false,  NULL, NULL);
  
    printf("Abracadabra\n");

    free(K);
    free(Beta);
    free(Gamma);
    free_pressure(&pressure);
    free_velocity_field(&Eta);
    free_velocity_field(&Zeta);
    free_velocity_field(&U);
    free_g_field(&g_field);

    return 0;
}
