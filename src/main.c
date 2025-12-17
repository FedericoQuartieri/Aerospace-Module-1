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
#include "forcing_parser.h"
#include "solve.h"

/* Solver for the Navier-Stokes-Brinkman equation */



int main(){
    // Initilize pressure
    Pressure pressure;
    initialize_pressure(&pressure);
    rand_fill(pressure.p);

    // Inizialize the 3 velocity field
    VelocityField Eta;
    VelocityField Zeta;
    VelocityField U;

    function v_boundary = parse_function("../v_boundary.txt");

    function forcing = parse_function("../forcing.txt");

    if (!forcing) {
        /* Error already printed to stderr */
        return 1;
    }


    initialize_velocity_field(&Eta, v_boundary);
    initialize_velocity_field(&Zeta, v_boundary);
    initialize_velocity_field(&U, v_boundary);
    // Set K 
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



    // Inizialize G
    GField g_field;
    initialize_g_field(&g_field);

    /** 
     * Compute G as: 
     *                 [dx]    f_x   - Grad_x(P) - c * U_x + c[ Grad_xx(N_x) + Grad_yy(Z_x) + Grad_zz(U_x)]
     *            G:   [dy] =  f_y   - Grad_y(P) - c * U_y + c[ Grad_xx(N_y) + Grad_yy(Z_y) + Grad_zz(U_y)]
     *                 [dz]    f_z   - Grad_z(P) - c * U_z + c[ Grad_xx(N_z) + Grad_yy(Z_z) + Grad_zz(U_z)] 
     * */     


    solve(g_field, forcing, pressure, K, Eta, Zeta, U, Beta, Gamma, v_boundary, 
        WRITE_FREQUENCY, false,  NULL, NULL);
  
    printf("momentum\n");

    destroy_forcing_function();


    free(K);
    free(Beta);
    free(Gamma);
    free_pressure(&pressure);
    free_velocity_field(&Eta);
    free_velocity_field(&Zeta);
    free_velocity_field(&U);
    free_g_field(&g_field);

    free(u_BC_current_direction);
    free(u_BC_derivative_second_direction);
    free(u_BC_derivative_third_direction);


    return 0;
}
