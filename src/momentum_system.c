#include "momentum_system.h"
#include <stdlib.h>
#include <write_vti_file.h>


void solve_momentum_system(VelocityField U, 
                           VelocityField Eta, 
                           VelocityField Zeta, 
                           Pressure *pressure_star,
                           VelocityField Xi,
                           GField g_field,
                           VelocityField Delta,
                           ForceField rhs,
                           DTYPE *Beta,
                           DTYPE *Gamma,          
                           const Data *data,
                           int timestep
                        )
{     
    //START(plane_Dxx);
    //compute_xi(g_field, U, Xi, Beta); // only needed for basic version of Dxx_tridiag
    compute_eta_next(Eta, Delta, Zeta, U, pressure_star, rhs, Xi, Gamma, data, timestep);
    //END_MS(plane_Dxx);
    //printf("Plane Dxx = %.3f ms\n", END_MS(plane_Dxx)); 

    compute_zeta_next(Zeta, Delta, rhs, Eta, Gamma, data, timestep);

    compute_u_next(U, Delta, rhs, Zeta, Gamma, data, timestep);
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
void compute_eta_next(VelocityField Eta, VelocityField Delta, VelocityField Zeta, VelocityField U, Pressure *pressure_star, ForceField rhs, VelocityField Xi, DTYPE *Gamma, const Data *data, int timestep){

    // rhs = Xi - Eta_n
/*     for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                rhs.f_x[idx] = Xi.v_x[idx] - Eta.v_x[idx];

                rhs.f_y[idx] = Xi.v_y[idx] - Eta.v_y[idx];

                rhs.f_z[idx] = Xi.v_z[idx] - Eta.v_z[idx];
            }
        }
    }  */ 

    /* 
        Warning: set the delta in the boundary conditions, both left and right,
        then update Eta also in the left points (idx=0)

        the Delta must be filled with the DELTA boundaries conditions, this means that since
        we are solving for the timestep t, then we set the delta_bound(t) = boundaries(t) - boundaries(t-1)
    */

    // Re-initialize Delta boundaries, modified by the previous system 
/*     update_delta_left_velocity_boundary(&Delta, timestep, data);
    update_delta_right_velocity_boundary(&Delta, timestep, data); */


    // Thomas algorithm for the linear system, for each component of Delta
/*      solve_Dxx_tridiag_blocks(Delta.v_x, rhs.f_x, Gamma, data, true, 0, timestep);
    solve_Dxx_tridiag_blocks(Delta.v_y, rhs.f_y, Gamma, data, false, 1, timestep);
    solve_Dxx_tridiag_blocks(Delta.v_z, rhs.f_z, Gamma, data, false, 2, timestep); 
  */

    START(optimize_tridiag_dxx);
    optimize_solve_Dxx_tridiag_blocks(Eta.v_x, Zeta.v_x, U.v_x, pressure_star->p, Gamma, data, true, 0, timestep);
    optimize_solve_Dxx_tridiag_blocks(Eta.v_y, Zeta.v_y, U.v_y, pressure_star->p, Gamma, data, false, 1, timestep);
    optimize_solve_Dxx_tridiag_blocks(Eta.v_z, Zeta.v_z, U.v_z, pressure_star->p, Gamma, data, false, 2, timestep);
    END_MS(optimize_tridiag_dxx);
    printf("Dxx_tridiag = %.3f ms\n", END_MS(optimize_tridiag_dxx));
 
    // Now in Delta we have the solution of the linear system: Delta = (Eta_n+1 - Eta_n)
    // we need to get Eta_n+1 as: Eta_n+1 = Delta + Eta_n
    // we have also the delta of the boundaries

    
    /*   for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                Eta.v_x[idx] = Delta.v_x[idx] + Eta.v_x[idx];

                Eta.v_y[idx] = Delta.v_y[idx] + Eta.v_y[idx]; 

                Eta.v_z[idx] = Delta.v_z[idx] + Eta.v_z[idx]; 
            }
        }
    }   */

}

/* (I - ∂yy) (Zeta_n+1 - Zeta_n) = Eta_n+1 - Zeta_n */
void compute_zeta_next(VelocityField Zeta, VelocityField Delta, ForceField rhs, VelocityField Eta, DTYPE *Gamma, const Data *data, int timestep){
    
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

    // Thomas algorithm for the linear system, for each component of Delta
/*       START(tridiag_dyy);
    solve_Dyy_tridiag_blocks(Delta.v_x, rhs.f_x, Gamma, data, false, 0, timestep);
    solve_Dyy_tridiag_blocks(Delta.v_y, rhs.f_y, Gamma, data, true, 1, timestep);
    solve_Dyy_tridiag_blocks(Delta.v_z, rhs.f_z, Gamma, data, false, 2, timestep);
   */

/*     START(simd_tridiag_dyy);
    vectorized_solve_Dyy_tridiag_blocks(Delta.v_x, rhs.f_x, Gamma, data, false, 0, timestep);
    vectorized_solve_Dyy_tridiag_blocks(Delta.v_y, rhs.f_y, Gamma, data, true, 1, timestep);
    vectorized_solve_Dyy_tridiag_blocks(Delta.v_z, rhs.f_z, Gamma, data, false, 2, timestep);
   */    
  
    START(optimized_simd_tridiag_dyy);
    optimized_simd_solve_Dyy_tridiag_blocks(Zeta.v_x, rhs.f_x, Gamma, data, false, 0, timestep);
    optimized_simd_solve_Dyy_tridiag_blocks(Zeta.v_y, rhs.f_y, Gamma, data, true, 1, timestep);
    optimized_simd_solve_Dyy_tridiag_blocks(Zeta.v_z, rhs.f_z, Gamma, data, false, 2, timestep);
    END_MS(optimized_simd_tridiag_dyy);
    printf("Dyy_tridiag_optimized_simd = %.3f ms\n", END_MS(optimized_simd_tridiag_dyy));
   


    // Now in Delta we have the solution of the linear system: Delta = (Zeta_n+1 - Zeta_n)
    // we need to get Zeta_n+1 as: Zeta_n+1 = Delta + Zeta_n
    /* !! Warning: trying to let left boundary as it (so to start from 1)!!*/
/*     for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                Zeta.v_x[idx] = Delta.v_x[idx] + Zeta.v_x[idx];

                Zeta.v_y[idx] = Delta.v_y[idx] + Zeta.v_y[idx]; 

                Zeta.v_z[idx] = Delta.v_z[idx] + Zeta.v_z[idx]; 
            }
        }
    }  */  

/*     END_MS(simd_tridiag_dyy);
    printf("Dyy_tridiag_simd = %.3f ms\n", END_MS(simd_tridiag_dyy)); */
 /*    END_MS(tridiag_dyy);
    printf("Dyy basic= %.3f ms\n", END_MS(tridiag_dyy));
   */
}

/* (I - ∂zz) (U_n+1 - U_n) = Zeta_n+1 - U_n */
void compute_u_next(VelocityField U, VelocityField Delta, ForceField rhs, VelocityField Zeta, DTYPE *Gamma, const Data *data, int timestep){
    
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

    // Thomas algorithm for the linear system, for each component of Delta
/*      START(tridiag_dzz);
    solve_Dzz_tridiag_blocks(Delta.v_x, rhs.f_x, Gamma, data, false, 0, timestep);
    solve_Dzz_tridiag_blocks(Delta.v_y, rhs.f_y, Gamma, data, false, 1, timestep);
    solve_Dzz_tridiag_blocks(Delta.v_z, rhs.f_z, Gamma, data, true, 2, timestep);
     */  
 

/*      START(simd_tridiag_dzz);
    vectorized_solve_Dzz_tridiag_blocks(Delta.v_x, rhs.f_x, Gamma, data, false, 0, timestep);
    vectorized_solve_Dzz_tridiag_blocks(Delta.v_y, rhs.f_y, Gamma, data, false, 1, timestep);
    vectorized_solve_Dzz_tridiag_blocks(Delta.v_z, rhs.f_z, Gamma, data, true, 2, timestep); 
     */ 

      START(optimized_simd_tridiag_dzz);
    optimized_simd_solve_Dzz_tridiag_blocks(U.v_x, rhs.f_x, Gamma, data, false, 0, timestep);
    optimized_simd_solve_Dzz_tridiag_blocks(U.v_y, rhs.f_y, Gamma, data, false, 1, timestep);
    optimized_simd_solve_Dzz_tridiag_blocks(U.v_z, rhs.f_z, Gamma, data, true, 2, timestep);
    END_MS(optimized_simd_tridiag_dzz);
    printf("Dzz_tridiag_optimized_simd = %.3f ms\n", END_MS(optimized_simd_tridiag_dzz));
     
 
    // Now in Delta we have the solution of the linear system: Delta = (U_n+1 - U_n)
    // we need to get U_n+1 as: U_n+1 = Delta + U_n
    /* !! Warning: trying to let left boundary as it (so to start from 1)!!*/
/*      for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                U.v_x[idx] = Delta.v_x[idx] + U.v_x[idx];

                U.v_y[idx] = Delta.v_y[idx] + U.v_y[idx]; 

                U.v_z[idx] = Delta.v_z[idx] + U.v_z[idx]; 
            }
        }
    } */ 

    //END_MS(tridiag_dzz);
    //printf("Dzz basic= %.3f ms\n", END_MS(tridiag_dzz));

    //END_MS(simd_tridiag_dzz);
    //printf("Dzz_tridiag = %.3f ms\n", END_MS(simd_tridiag_dzz));

    
}

/**
 * Compute Xi for the three components x,y,z
 * Xi_n+1 = U_n + (dt/β) * g_n
 *  */
void compute_xi(GField g_field, VelocityField U, VelocityField Xi, DTYPE *Beta){
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);
                
                DTYPE coeff = DT / Beta[idx];

                Xi.v_x[idx] = U.v_x[idx] + coeff * g_field.g_x[idx];

                Xi.v_y[idx] = U.v_y[idx] + coeff * g_field.g_y[idx];

                Xi.v_z[idx] = U.v_z[idx] + coeff * g_field.g_z[idx];    
            }
        }
    }
}

