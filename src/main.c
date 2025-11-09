#include <stdio.h>
#include <stdlib.h>
#include "../include/constants.h"
#include "velocity_field.h"
#include "pressure.h"
#include "force_field.h"
#include "g_field.h"
#include "utils.h"
#include "momentum_system.h"

/* Solver for the Navier-Stokes-Brinkman equation */

int main(){

    // Initialize Force field
    ForceField f_field;
    initialize_force_field(&f_field);

    // Initilize pressure
    Pressure pressure;
    initialize_pressure(&pressure);
    rand_fill(pressure.p);

    // Inizialize the 3 velocity field
    VelocityField Eta;
    VelocityField Zeta;
    VelocityField U;
    initialize_velocity_field(&Eta);
    initialize_velocity_field(&Zeta);
    initialize_velocity_field(&U);

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

    // Inizialize G
    GField g_field;
    initialize_g_field(&g_field);

    /** 
     * Compute G as: 
     *                 [dx]    f_x   - Grad_x(P) - c * U_x + c[ Grad_xx(N_x) + Grad_yy(Z_x) + Grad_zz(U_x)]
     *            G:   [dy] =  f_y   - Grad_y(P) - c * U_y + c[ Grad_xx(N_y) + Grad_yy(Z_y) + Grad_zz(U_y)]
     *                 [dz]    f_z   - Grad_z(P) - c * U_z + c[ Grad_xx(N_z) + Grad_yy(Z_z) + Grad_zz(U_z)] 
     * */     
    compute_g(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U);
    


    // For each time step:
    // declare U_next, Eta_next, Zeta_next, Xi
        VelocityField Eta_next;
        VelocityField Zeta_next;
        VelocityField U_next;
        VelocityField Xi;
        initialize_velocity_field(&Xi);
        initialize_velocity_field(&Eta_next);
        initialize_velocity_field(&Zeta_next);
        initialize_velocity_field(&U_next);
        solve_momentum_system(U, Eta, Zeta, Xi, g_field, K, U_next, Eta_next, Zeta_next, Beta, Gamma);

    printf("momentum\n");

    free(K);
    free_force_field(&f_field);
    free_pressure(&pressure);
    free_velocity_field(&Eta);
    free_velocity_field(&Zeta);
    free_velocity_field(&U);
    free_g_field(&g_field);

    free_velocity_field(&Xi);
    free_velocity_field(&Eta_next);
    free_velocity_field(&Zeta_next);
    free_velocity_field(&U_next);

    return 0;
}
