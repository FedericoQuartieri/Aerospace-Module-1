#include "momentum_system.h"

/*
    Updated by Ema, 

    Delta is the velocity_field now used to store the solution of the linear system,
    and the solution of the n+1 step is updated directly into Eta,Zeta,U.
    The _next was not necessary, now we just use the Delta field to store the solution 
    and then update the associated velocity.
    
    The same thing applied to the (previously f_field) rhs, since now is initialized once in the solve.c
    and passed at each timestep to this file.
    It's also shared among the 3 system, to reduce the number of malloc/free and improve memory usage.

    Note that since in Delta the boundary values are used in the Thomas algorithm, and for the 
    'different_direction' method they are changed by the algorithm, it's necessary that before
    each system the boundary in the Delta field are re-initialize (also inside the same timestep).
*/
void solve_momentum_system(VelocityField U, 
                           VelocityField Eta, 
                           VelocityField Zeta, 
                           VelocityField Xi,
                           GField g_field,
                           VelocityField Delta,
                           ForceField rhs,
                           DTYPE *Beta,
                           DTYPE *Gamma,
                           function_handle v_boundary,
                           int timestep
                        )
{     
    compute_xi(g_field, U, Xi, Beta);

    compute_eta_next(Eta, Delta, rhs, Xi, Gamma, v_boundary, timestep);

    compute_zeta_next(Zeta, Delta, rhs, Eta, Gamma, v_boundary, timestep);

    compute_u_next(U, Delta, rhs, Zeta, Gamma, v_boundary, timestep);
}

/* 
    Some doubts... (OLD , to be deleted?)

    After computing Delta with Thomas, we compute the value of V_n+1 as = V_n + Delta
    1) The left boundary values, shouldn't be summed with previous. So the update will now start from 1
    
    2) The right boundary values instead are computed with Thomas as the other points, so it should be
        correct sum them with the older i think...
    
    3) At the end of each equations, we have the final solution for its boundary faces,
        becouse for each equation we compute correctly 2 buondary face.
        However in our corrently solution, we are 'throwing' them, since in the last equation with U,
        we compute Thomas also on the other 4 faces, and take only the z-faces as boundary.
        I would like to try to store for each equation, the value of the associated boundary faces, in order 
        to put them into U after all.
        (Maybe also avoiding thomas on useless faces for each equation)
*/

/* (I - ∂xx) (Eta_n+1 - Eta_n) = Xi - Eta_n */
static void compute_eta_next(VelocityField Eta, VelocityField Delta, ForceField rhs, VelocityField Xi, DTYPE *Gamma, function_handle v_boundary, int timestep){

    // rhs = Xi - Eta_n
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

    /* 
        Warning: set the delta in the boundary conditions, both left and right,
        then update Eta also in the left points (idx=0)

        the Delta must be filled with the DELTA boundaries conditions, this means that since
        we are solving for the timestep t, then we set the delta_bound(t) = boundaries(t) - boundaries(t-1)
    */

    // Re-initialize Delta boundaries, modified by the previous system 
    update_delta_left_velocity_boundary(&Delta, v_boundary, timestep);
    update_delta_right_velocity_boundary(&Delta, v_boundary, timestep);

    // Thomas algorithm for the linear system, for each component of Delta
    solve_Dxx_tridiag_blocks(Delta.v_x, rhs.f_x, Gamma, v_boundary, true);
    solve_Dxx_tridiag_blocks(Delta.v_y, rhs.f_y, Gamma, v_boundary, false);
    solve_Dxx_tridiag_blocks(Delta.v_z, rhs.f_z, Gamma, v_boundary, false);

    // Now in Delta we have the solution of the linear system: Delta = (Eta_n+1 - Eta_n)
    // we need to get Eta_n+1 as: Eta_n+1 = Delta + Eta_n

    /* !! Warning: since we are imposing the delta of the boundaries, we should start from idx=0 !!*/
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                Eta.v_x[idx] = Delta.v_x[idx] + Eta.v_x[idx];

                Eta.v_y[idx] = Delta.v_y[idx] + Eta.v_y[idx]; 

                Eta.v_z[idx] = Delta.v_z[idx] + Eta.v_z[idx]; 
            }
        }
    }

}

/* (I - ∂yy) (Zeta_n+1 - Zeta_n) = Eta_n+1 - Zeta_n */
static void compute_zeta_next(VelocityField Zeta, VelocityField Delta, ForceField rhs, VelocityField Eta, DTYPE *Gamma, function_handle v_boundary, int timestep){
    
    // rhs = Eta_n+1 - Zeta_n
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                rhs.f_x[idx] = Eta.v_x[idx] - Zeta.v_x[idx];

                rhs.f_y[idx] = Eta.v_y[idx] - Zeta.v_y[idx];

                rhs.f_z[idx] = Eta.v_z[idx] - Zeta.v_z[idx];
            }
        }
    }

    // Re-initialize Delta boundaries, modified by the previous system 
    update_delta_left_velocity_boundary(&Delta, v_boundary, timestep);
    update_delta_right_velocity_boundary(&Delta, v_boundary, timestep);

    // Thomas algorithm for the linear system, for each component of Delta
    solve_Dyy_tridiag_blocks(Delta.v_x, rhs.f_x, Gamma, v_boundary, false);
    solve_Dyy_tridiag_blocks(Delta.v_y, rhs.f_y, Gamma, v_boundary, true);
    solve_Dyy_tridiag_blocks(Delta.v_z, rhs.f_z, Gamma, v_boundary, false);

    // Now in Delta we have the solution of the linear system: Delta = (Zeta_n+1 - Zeta_n)
    // we need to get Zeta_n+1 as: Zeta_n+1 = Delta + Zeta_n
    /* !! Warning: trying to let left boundary as it (so to start from 1)!!*/
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                Zeta.v_x[idx] = Delta.v_x[idx] + Zeta.v_x[idx];

                Zeta.v_y[idx] = Delta.v_y[idx] + Zeta.v_y[idx]; 

                Zeta.v_z[idx] = Delta.v_z[idx] + Zeta.v_z[idx]; 
            }
        }
    }

}

/* (I - ∂zz) (U_n+1 - U_n) = Zeta_n+1 - U_n */
static void compute_u_next(VelocityField U, VelocityField Delta, ForceField rhs, VelocityField Zeta, DTYPE *Gamma, function_handle v_boundary, int timestep){
    
    // rhs = Zeta_n+1 - U_n
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                rhs.f_x[idx] = Zeta.v_x[idx] - U.v_x[idx];

                rhs.f_y[idx] = Zeta.v_y[idx] - U.v_y[idx];

                rhs.f_z[idx] = Zeta.v_z[idx] - U.v_z[idx];
            }
        }
    }

    // Re-initialize Delta boundaries, modified by the previous system 
    update_delta_left_velocity_boundary(&Delta, v_boundary, timestep);
    update_delta_right_velocity_boundary(&Delta, v_boundary, timestep);

    // Thomas algorithm for the linear system, for each component of Delta
    solve_Dzz_tridiag_blocks(Delta.v_x, rhs.f_x, Gamma, v_boundary, false);
    solve_Dzz_tridiag_blocks(Delta.v_y, rhs.f_y, Gamma, v_boundary, false);
    solve_Dzz_tridiag_blocks(Delta.v_z, rhs.f_z, Gamma, v_boundary, true);

    // Now in Delta we have the solution of the linear system: Delta = (U_n+1 - U_n)
    // we need to get U_n+1 as: U_n+1 = Delta + U_n

    /* !! Warning: trying to let left boundary as it (so to start from 1)!!*/
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                U.v_x[idx] = Delta.v_x[idx] + U.v_x[idx];

                U.v_y[idx] = Delta.v_y[idx] + U.v_y[idx]; 

                U.v_z[idx] = Delta.v_z[idx] + U.v_z[idx]; 
            }
        }
    }
    
}

/**
 * Compute Xi for the three components x,y,z
 * Xi_n+1 = U_n + (dt/β) * g_n
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

