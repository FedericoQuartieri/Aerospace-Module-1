#include "momentum_system.h"

void solve_momentum_system(VelocityField U, 
                           VelocityField Eta, 
                           VelocityField Zeta, 
                           VelocityField Xi,
                           GField g_field,
                           DTYPE *K,
                           VelocityField U_next,
                           VelocityField Eta_next,
                           VelocityField Zeta_next,
                           DTYPE *Beta,
                           DTYPE *Gamma,
                           VelocityField vx_bound,
                           VelocityField vy_bound,
                           VelocityField vz_bound,)
{
    compute_xi(g_field, U, Xi, Beta);
    compute_eta_next(Eta, Eta_next, Xi, Gamma,
                            vx_bound,
                            vy_bound, vz_bound);
    compute_zeta_next(Zeta, Zeta_next, Eta_next, Gamma,
                            vx_bound,
                            vy_bound, vz_bound);
    compute_u_next(U, U_next, Zeta_next, Gamma,
                            vx_bound,
                            vy_bound, vz_bound);
}

/* (I - ∂xx) (Eta_next - Eta) = Xi - Eta */
static void compute_eta_next(VelocityField Eta, VelocityField Eta_next, VelocityField Xi, DTYPE *Gamma,    
                            VelocityField vx_bound,
                            VelocityField vy_bound, VelocityField vz_bound){
    // Right-hand side for the tridiagonal system
    ForceField rhs;
    initialize_force_field(&rhs);

    //check se serve iterare sui ghost
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                rhs.f_x[idx] = Xi.v_x[idx] - Eta.v_x[idx];

                rhs.f_y[idx] = Xi.v_y[idx] - Eta.v_y[idx];

                rhs.f_z[idx] = Xi.v_z[idx] - Eta.v_z[idx];
            }
        }
    }

    // Thomas algorithm for the linear system, for each component of Eta_next
    solve_Dxx_tridiag_blocks(Eta_next.v_x, rhs.f_x, Gamma,
                            vx_bound, true);
    solve_Dxx_tridiag_blocks(Eta_next.v_y, rhs.f_y, Gamma,
                            vy_bound, false);
    solve_Dxx_tridiag_blocks(Eta_next.v_z, rhs.f_z, Gamma,
                            vz_bound, false);
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
    
    free_force_field(&rhs);
}

/* (I - ∂yy) (Zeta_next - Zeta) = Eta_next - Zeta */
static void compute_zeta_next(VelocityField Zeta, VelocityField Zeta_next, VelocityField Eta_next, DTYPE *Gamma,
                            VelocityField vx_bound,
                            VelocityField vy_bound, VelocityField vz_bound){
    // Right-hand side for the tridiagonal system
    ForceField rhs;
    initialize_force_field(&rhs);

    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                rhs.f_x[idx] = Eta_next.v_x[idx] - Zeta.v_x[idx];

                rhs.f_y[idx] = Eta_next.v_y[idx] - Zeta.v_y[idx];

                rhs.f_z[idx] = Eta_next.v_z[idx] - Zeta.v_z[idx];
            }
        }
    }

    // Thomas algorithm for the linear system, for each component of Zeta_next
    solve_Dyy_tridiag_blocks(Zeta_next.v_x, rhs.f_x, Gamma,
                            vx_bound, false);
    solve_Dyy_tridiag_blocks(Zeta_next.v_y, rhs.f_y, Gamma,
                            vy_bound, true);
    solve_Dyy_tridiag_blocks(Zeta_next.v_z, rhs.f_z, Gamma,
                            vz_bound, false);
    // Now in Eta_next we have the solution of the linear system: s = (Eta_next - Eta)
    // we need to get Eta_next as: Eta_next = s + Eta
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                Zeta_next.v_x[idx] = Zeta_next.v_x[idx] + Zeta.v_x[idx];

                Zeta_next.v_y[idx] = Zeta_next.v_y[idx] + Zeta.v_y[idx]; 

                Zeta_next.v_z[idx] = Zeta_next.v_z[idx] + Zeta.v_z[idx]; 
            }
        }
    }
    
    free_force_field(&rhs);
}

/* (I - ∂zz) (U_next - U) = Zeta_next - U */
static void compute_u_next(VelocityField U, VelocityField U_next, VelocityField Zeta_next, DTYPE *Gamma,
                            VelocityField vx_bound,
                            VelocityField vy_bound, VelocityField vz_bound){
    // Right-hand side for the tridiagonal system
    ForceField rhs;
    initialize_force_field(&rhs);

    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                rhs.f_x[idx] = Zeta_next.v_x[idx] - U.v_x[idx];

                rhs.f_y[idx] = Zeta_next.v_y[idx] - U.v_y[idx];

                rhs.f_z[idx] = Zeta_next.v_z[idx] - U.v_z[idx];
            }
        }
    }

    // Thomas algorithm for the linear system, for each component of Zeta_next
    solve_Dzz_tridiag_blocks(U_next.v_x, rhs.f_x, Gamma,
                            vx_bound, false);
    solve_Dzz_tridiag_blocks(U_next.v_y, rhs.f_y, Gamma,
                            vy_bound, false);
    solve_Dzz_tridiag_blocks(U_next.v_z, rhs.f_z, Gamma,
                            vz_bound, true);
    // Now in Eta_next we have the solution of the linear system: s = (Eta_next - Eta)
    // we need to get Eta_next as: Eta_next = s + Eta
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                U_next.v_x[idx] = U_next.v_x[idx] + U.v_x[idx];

                U_next.v_y[idx] = U_next.v_y[idx] + U.v_y[idx]; 

                U_next.v_z[idx] = U_next.v_z[idx] + U.v_z[idx]; 
            }
        }
    }
    
    free_force_field(&rhs);
}

/**
 * Compute Xi for the three components x,y,z
 * xi_n+1 = u_n + (dt/β) * g_n
 *  */
static void compute_xi(GField g_field, VelocityField U, VelocityField Xi, DTYPE *Beta){
    for(int k = 1; k < DEPTH; k++){
        for(int j = 1; j < HEIGHT; j++){
            for(int i = 1; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);
                
                DTYPE coeff = DT / Beta[idx];

                Xi.v_x[idx] = U.v_x[idx] + coeff * g_field.g_x[idx];

                Xi.v_y[idx] = U.v_y[idx] + coeff * g_field.g_y[idx];

                Xi.v_z[idx] = U.v_z[idx] + coeff * g_field.g_z[idx];
            }
        }
    }
}

