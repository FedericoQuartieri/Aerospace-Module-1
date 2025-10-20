#include <stdio.h>
#include "momentum_system.h"

void solve_momentum_system(VelocityField U, 
                           VelocityField Eta, 
                           VelocityField Zeta, 
                           VelocityField Xi,
                           GField g_field,
                           DTYPE *K,
                           VelocityField U_next,
                           VelocityField Eta_next,
                           VelocityField Zeta_next)
{
    compute_xi(g_field, U, Xi, K);
    compute_eta_next(Eta, Eta_next, Xi, K);
    compute_zeta_next(Zeta, Zeta_next, Eta_next, K);
    compute_u_next(U, U_next, Zeta_next, K);
}

static void compute_eta_next(VelocityField Eta, VelocityField Eta_next, VelocityField Xi, DTYPE *K){


}

/**
 * Compute xi for the three components x,y,z
 * xi_n+1 = u_n + (dt/β) * g_n
 *  */
static void compute_xi(GField g_field, VelocityField U, VelocityField Xi, DTYPE *K){
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                DTYPE beta = 1 + (DT * NU) / (2 * K[idx]); 
                DTYPE coeff = DT / beta;

                Xi.v_x[idx] = U.v_x[idx] + coeff * g_field.g_x[idx];

                Xi.v_y[idx] = U.v_y[idx] + coeff * g_field.g_y[idx];

                Xi.v_z[idx] = U.v_z[idx] + coeff * g_field.g_z[idx];
            }
        }
    }
}

