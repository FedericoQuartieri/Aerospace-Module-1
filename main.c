#include <stdio.h>
#include <stdlib.h>
#include "constants.h"
#include "velocity_field.h"
#include "pressure.h"
#include "force_field.h"
#include "g_field.h"
   
/* Solver for the Navier-Stokes-Brinkman equation */

int main(){

    // Initialize Force field
    ForceField f_field;
    initialize_force_field(&f_field);

    // Initilize pressure
    Pressure pressure;
    initialize_pressure(&pressure);

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
    
    printf("Done?\n");

    free(K);
    free_force_field(&f_field);
    free_pressure(&pressure);
    free_velocity_field(&Eta);
    free_velocity_field(&Zeta);
    free_velocity_field(&U);
    free_g_field(&g_field);

    return 0;
}












