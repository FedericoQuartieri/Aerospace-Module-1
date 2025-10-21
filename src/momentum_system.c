#include "momentum_system.h"

static void compute_eta_next(VelocityField Eta, VelocityField Eta_next, VelocityField Xi, DTYPE *K);
static void compute_xi(GField g_field, VelocityField U, VelocityField Xi, DTYPE *K);

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
    //compute_zeta_next(Zeta, Zeta_next, Eta_next, K);
    //compute_u_next(U, U_next, Zeta_next, K);
}

/* (I - ∂xx) (Eta_next - Eta) = Xi - Eta */
static void compute_eta_next(VelocityField Eta, VelocityField Eta_next, VelocityField Xi, DTYPE *K){
    // Right-hand side for the tridiagonal system
    ForceField f_field;
    initialize_force_field(&f_field);
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                f_field.f_x[idx] = Xi.v_x[idx] - Eta.v_x[idx];

                f_field.f_y[idx] = Xi.v_y[idx] - Eta.v_y[idx];

                f_field.f_z[idx] = Xi.v_z[idx] - Eta.v_z[idx];
            }
        }
    }

    // Thomas algorithm for the linear system, for each component of Eta_next
    solve_Dxx_tridiag_blocks(Eta_next.v_x, f_field.f_x);
    solve_Dxx_tridiag_blocks(Eta_next.v_y, f_field.f_y);
    solve_Dxx_tridiag_blocks(Eta_next.v_z, f_field.f_z);

    // Now in Eta_next we have the solution of the linear system: s = (Eta_next - Eta)
    // we need to get Eta_next as: Eta_next = s + Eta
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                Eta_next.v_x[idx] = Eta_next.v_x[idx] + Eta.v_x[idx];

                Eta_next.v_y[idx] = Eta_next.v_y[idx] + Eta.v_y[idx]; 

                Eta_next.v_z[idx] = Eta_next.v_z[idx] + Eta.v_z[idx]; 
            }
        }
    }
}

/**
 * Compute Xi for the three components x,y,z
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

