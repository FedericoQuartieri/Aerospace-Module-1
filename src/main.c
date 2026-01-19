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
#include "function.h"
#include "solve.h"

/* Navier-Stokes-Brinkman equation solver */

int main(){
    
    Pressure pressure;
    initialize_pressure(&pressure);
    rand_fill(pressure.p);

    VelocityField Eta;
    VelocityField Zeta;
    VelocityField U;

    function_handle v_boundary = parse_function("../v_boundary.txt");

    function_handle forcing = parse_function("../forcing.txt");

    if (!forcing) {
        /* Error already printed to stderr */
        return 1;
    }

    // Initilized 3 velocity field, and for each one set the SAME boundary conditions,
    // Need to analyze performance...
    initialize_velocity_field(&Eta, v_boundary);
    initialize_velocity_field(&Zeta, v_boundary);
    initialize_velocity_field(&U, v_boundary);

    // Set K that is needed to compute Gamma
    // K depends also on the spatial coordinate
    DTYPE *K = (DTYPE *) malloc(GRID_SIZE);
    rand_fill(K); 
    
    DTYPE *Beta = (DTYPE *) malloc(GRID_SIZE);
    DTYPE *Gamma = (DTYPE *) malloc(GRID_SIZE);
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);
                Beta[idx] = 1 + (DT * NU) / (2 * K[idx]);
                Gamma[idx] = (DT * NU) / ( 2 * Beta[idx]);
            }
        }
    }
    
    double x, y, z, t;
    double fx, fy, fz;

    // Inizialize g
    GField g_field;
    initialize_g_field(&g_field);    

    solve(g_field, forcing, pressure, K, Eta, Zeta, U, Beta, Gamma, v_boundary, 
        WRITE_FREQUENCY, false,  NULL, NULL);
  
    printf("Abracadabra\n");

    destroy_function(v_boundary);
    destroy_function(forcing);

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
