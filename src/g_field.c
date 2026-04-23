#include "g_field.h"
#include "data.h"

void initialize_g_field(GField *g_field){
    g_field->g_x = (DTYPE *) malloc(GRID_SIZE);
    g_field->g_y = (DTYPE *) malloc(GRID_SIZE);
    g_field->g_z = (DTYPE *) malloc(GRID_SIZE);

    memset(g_field->g_x, 0, GRID_SIZE);
    memset(g_field->g_y, 0, GRID_SIZE);
    memset(g_field->g_z, 0, GRID_SIZE);
}

void free_g_field(GField *g_field){
    free(g_field->g_x);
    free(g_field->g_y);
    free(g_field->g_z);
}

/* 
    Update by Ema:

    We might want to pass to eval_function() the physical coordinates (i.e. i*DX, j*DY, k*DZ) instead of 
    the index in the grid.
    This is done since we have a physical domain [0, L]^3 and a grid built as:
    DX = L/(WIDTH-1), DY = L/(HEIGHT-1), DZ = L/(DEPTH-1), DT = Total_time/Steps

    So to get the physical coordinates from the index of the grid:
    x = i * DX , y = j * DY, z = k * DZ, t = time_step * DT

    -> Now that we have the physical coordinate of the pressure point, since we are using
    a staggered grid (where velocity point are shifted of D/2), we then set:
    v_x (coordinate) = (x + DX/2, y, z)
    v_y (coordinate) = (x, y + DY/2, z)
    v_z (coordinate) = (x, y, z + DZ/2)

*/
/*
    To compute the laplacian for boundary components such that:

            |
            |
   *-->--*-->--*-->
            |
   N-1   N  |  N+1 (ghost node)

   (v_N+1 + v_N-1) / 2 = v_N  -> v_N+1 = 2v_N - v_N-1

   laplacian v_N ->  v_N-1 -2v_N + v_N+1 
   
   Now substituting v_N+1 with this definition, we have:

   v_N-1 -2v_N + (2v_n - v_N-1) = 0

   So the contribution of the laplacian of components on the boundary is 0
---------------------------------------------------------------------------
    To compute the laplacian for boundary components such that:

           |
   ^    ^  |  ^  
   *----*--|--*--
           |
  N-1   N  |  N+1 (ghost node)

   (v_N+1 + v_N) / 2 = v_ex  -> v_N+1 = 2v_ex - v_N

   laplacian v_N ->  v_N-1 -2v_N + v_N+1 
   
   Now substituting v_N+1 with this definition, we have:

   v_N-1 -2v_N + (2v_ex - v_N) 

*/
/*
    Initialize solver at t=0 (t-1), starting the first timpestep at t=1 (t):
    g(t-1/2) = f(t-1/2) + p(t-1/2) - u(t-1) + (laplacian[Eta(t-1) + Zeta(t-1) + U(t-1)])

    this means that:
    - forcing term is evaluated in the timestep t-1/2
    - velocity term are evaluated in t-1
*/
void compute_g(GField *g_field, Pressure *pressure, DTYPE *K, VelocityField *Eta, VelocityField *Zeta, VelocityField *U, int time_step, const Data *data){

    size_t idx, left_idx, down_idx, back_idx;
    // Physical coordinates of the domain
    DTYPE t_half_prev = (time_step - 0.5) * DT;
    DTYPE t_prev = (time_step - 1) * DT; 
    DTYPE x_max = (WIDTH - 1) * DX;
    DTYPE y_max = (HEIGHT - 1) * DY;
    DTYPE z_max = (DEPTH - 1) * DZ;
    DTYPE vel_x_max = x_max + DX/2;
    DTYPE vel_y_max = y_max + DY/2;
    DTYPE vel_z_max = z_max + DZ/2;
    
    for(int k = 1; k < DEPTH-1; k++){
        for(int j = 1; j < HEIGHT-1; j++){
            for(int i = 1; i < WIDTH-1; i++){
                
                idx = rowmaj_idx(i,j,k);
                DTYPE x = i * DX;
                DTYPE y = j * DY;
                DTYPE z = k * DZ;
                DTYPE vel_x = x + DX/2;
                DTYPE vel_y = y + DY/2;
                DTYPE vel_z = z + DZ/2;

                g_field->g_x[idx] = forcing(data, vel_x, y, z, t_half_prev, 0)
                                    - compute_pressure_x_grad(pressure->p,i,j,k) 
                                    - (NU / (K[idx])) * U->v_x[idx]
                                    + (NU) * (compute_velocity_xx_grad(Eta->v_x,i,j,k)  
                                                + compute_velocity_yy_grad(Zeta->v_x,i,j,k) 
                                                + compute_velocity_zz_grad(U->v_x,i,j,k));
                                    

                g_field->g_y[idx] =  forcing(data, x, vel_y, z, t_half_prev, 1)
                                    - compute_pressure_y_grad(pressure->p,i,j,k) 
                                    - (NU / (K[idx])) * U->v_y[idx]
                                    + (NU) * (compute_velocity_xx_grad(Eta->v_y,i,j,k)  
                                                + compute_velocity_yy_grad(Zeta->v_y,i,j,k) 
                                                + compute_velocity_zz_grad(U->v_y,i,j,k)); 
                 

                g_field->g_z[idx] =  forcing(data, x, y, vel_z, t_half_prev, 2)
                                    - compute_pressure_z_grad(pressure->p,i,j,k) 
                                    - (NU / (K[idx])) * U->v_z[idx]
                                    + (NU) * (compute_velocity_xx_grad(Eta->v_z,i,j,k)  
                                                + compute_velocity_yy_grad(Zeta->v_z,i,j,k) 
                                                + compute_velocity_zz_grad(U->v_z,i,j,k));
            }
        }
    }

    /* Assuming homogeneous Neumann condition, then pressure gradient is 0 in right boundary */ 

    // (i,j,k) = (WIDTH-1,HEIGHT-1,DEPTH-1)
    idx = rowmaj_idx(WIDTH-1,HEIGHT-1,DEPTH-1);
    left_idx = rowmaj_idx(WIDTH-2,HEIGHT-1,DEPTH-1);
    down_idx = rowmaj_idx(WIDTH-1,HEIGHT-2,DEPTH-1);
    back_idx = rowmaj_idx(WIDTH-1,HEIGHT-1,DEPTH-2);

    // (i,j,k) = (WIDTH-1,HEIGHT-1,k)
    for(int k = 1; k < DEPTH - 1; k++){
        idx = rowmaj_idx(WIDTH-1,HEIGHT-1,k);
        left_idx = rowmaj_idx(WIDTH-2,HEIGHT-1,k);
        down_idx = rowmaj_idx(WIDTH-1,HEIGHT-2,k);
        DTYPE z = k * DZ;
        DTYPE vel_z = z + DZ/2;

        // This is the ONLY that propagates to the solution U 
        g_field->g_z[idx] = forcing(data, x_max, y_max, vel_z, t_half_prev, 2) 
                    - compute_pressure_z_grad(pressure->p,WIDTH-1,HEIGHT-1,k) 
                    - (NU / (K[idx])) * U->v_z[idx]
                    + (NU) * (((Eta->v_z[left_idx] - 2.0*Eta->v_z[idx] + (2.0 * bc_velocity(data, vel_x_max, y_max, vel_z, t_prev, 2) - Eta->v_z[idx])) * DX_INVERSE_SQUARE)    
                                + ((Zeta->v_z[down_idx] - 2.0*Zeta->v_z[idx] + (2.0 * bc_velocity(data, x_max, vel_y_max, vel_z, t_prev, 2) - Zeta->v_z[idx])) * DY_INVERSE_SQUARE) 
                                + compute_velocity_zz_grad(U->v_z,WIDTH-1,HEIGHT-1,k) ); 
    }   

    // (i,j,k) = (WIDTH-1,j,DEPTH-1)
    for(int j = 1; j < HEIGHT - 1; j++){
        idx = rowmaj_idx(WIDTH-1,j,DEPTH-1);
        left_idx = rowmaj_idx(WIDTH-2,j,DEPTH-1);
        back_idx = rowmaj_idx(WIDTH-1,j,DEPTH-2);
        DTYPE y = j * DY;
        DTYPE vel_y = y + DY/2;

        // This will propagate to the solution U
        g_field->g_y[idx] = forcing(data, x_max, vel_y, z_max, t_half_prev, 1) 
                    - compute_pressure_y_grad(pressure->p,WIDTH-1,j,DEPTH-1) 
                    - (NU / (K[idx])) * U->v_y[idx]
                    + (NU) * (((Eta->v_y[left_idx] - 2.0*Eta->v_y[idx] + (2.0 * bc_velocity(data, vel_x_max, vel_y, z_max, t_prev, 1) - Eta->v_y[idx])) * DX_INVERSE_SQUARE)  
                                + compute_velocity_yy_grad(Zeta->v_y,WIDTH-1,j,DEPTH-1) 
                                + ((U->v_y[back_idx] - 2.0*U->v_y[idx] + (2.0 * bc_velocity(data, x_max, vel_y, vel_z_max, t_prev, 1) - U->v_y[idx])) * DZ_INVERSE_SQUARE) );

  }

    // (i,j,k) = (i,HEIGHT-1,DEPTH-1)
    for(int i = 1; i < WIDTH - 1; i++){
        idx = rowmaj_idx(i,HEIGHT-1,DEPTH-1);
        down_idx = rowmaj_idx(i,HEIGHT-2,DEPTH-1);
        back_idx = rowmaj_idx(i,HEIGHT-1,DEPTH-2);
        DTYPE x = i * DX;
        DTYPE vel_x = x + DX/2;

        // This will propagate to the solution U
        g_field->g_x[idx] = forcing(data, vel_x, y_max, z_max, t_half_prev, 0) 
                    - compute_pressure_x_grad(pressure->p,i,HEIGHT-1,DEPTH-1) 
                    - (NU / (K[idx])) * U->v_x[idx]
                    + (NU) * (compute_velocity_xx_grad(Eta->v_x,i,HEIGHT-1,DEPTH-1)  
                                + ((Zeta->v_x[down_idx] - 2.0*Zeta->v_x[idx] + (2.0 * bc_velocity(data, vel_x, vel_y_max, z_max, t_prev, 0) - Zeta->v_x[idx])) * DY_INVERSE_SQUARE) 
                                + ((U->v_x[back_idx] - 2.0*U->v_x[idx] + (2.0 * bc_velocity(data, vel_x, y_max, vel_z_max, t_prev, 0) - U->v_x[idx])) * DZ_INVERSE_SQUARE) );
        
    }

    // (i, j, k) = (WIDTH-1,j,k)
    for(int k = 1; k < DEPTH - 1; k++){
        for(int j = 1; j < HEIGHT - 1; j++){
            idx = rowmaj_idx(WIDTH-1,j,k);
            left_idx = rowmaj_idx(WIDTH-2,j,k);
            DTYPE y = j * DY;
            DTYPE z = k * DZ;
            DTYPE vel_y = y + DY/2;
            DTYPE vel_z = z + DZ/2;

            // This will propagate to the solution U
            g_field->g_y[idx] = forcing(data, x_max, vel_y, z, t_half_prev, 1) 
                        - compute_pressure_y_grad(pressure->p,WIDTH-1,j,k) 
                        - (NU / (K[idx])) * U->v_y[idx]
                        + (NU) * (((Eta->v_y[left_idx] - 2.0*Eta->v_y[idx] + (2.0 * bc_velocity(data, vel_x_max, vel_y, z, t_prev, 1) - Eta->v_y[idx])) * DX_INVERSE_SQUARE)  
                                    + compute_velocity_yy_grad(Zeta->v_y,WIDTH-1,j,k) 
                                    + compute_velocity_zz_grad(U->v_y,WIDTH-1,j,k) );
            
            // This will propagate to the solution U
            g_field->g_z[idx] = forcing(data, x_max, y, vel_z, t_half_prev, 2) 
                        - compute_pressure_z_grad(pressure->p,WIDTH-1,j,k) 
                        - (NU / (K[idx])) * U->v_z[idx]
                        + (NU) * (((Eta->v_z[left_idx] - 2.0*Eta->v_z[idx] + (2.0 * bc_velocity(data, vel_x_max, y, vel_z, t_prev, 2) - Eta->v_z[idx])) * DX_INVERSE_SQUARE)  
                                    + compute_velocity_yy_grad(Zeta->v_z,WIDTH-1,j,k) 
                                    + compute_velocity_zz_grad(U->v_z,WIDTH-1,j,k) );
         
        }
    }

    //(i, j, k) = (i,HEIGHT-1,k)
    for(int k = 1; k < DEPTH - 1; k++){
        for(int i = 1; i < WIDTH - 1; i++){
            idx = rowmaj_idx(i,HEIGHT-1,k);
            down_idx = rowmaj_idx(i,HEIGHT-2,k);
            DTYPE x = i * DX;
            DTYPE z = k * DZ;
            DTYPE vel_x = x + DX/2;
            DTYPE vel_z = z + DZ/2;

            // This will propagate to the solution U
            g_field->g_x[idx] = forcing(data, vel_x, y_max, z, t_half_prev, 0) 
                        - compute_pressure_x_grad(pressure->p,i,HEIGHT-1,k) 
                        - (NU / (K[idx])) * U->v_x[idx]
                        + (NU) * (compute_velocity_xx_grad(Eta->v_x,i,HEIGHT-1,k)  
                                    + ((Zeta->v_x[down_idx] - 2.0*Zeta->v_x[idx] + (2.0 * bc_velocity(data, vel_x, vel_y_max, z, t_prev, 0) - Zeta->v_x[idx])) * DY_INVERSE_SQUARE) 
                                    + compute_velocity_zz_grad(U->v_x,i,HEIGHT-1,k) );

            // This will propagate to the solution U
            g_field->g_z[idx] = forcing(data, x, y_max, vel_z, t_half_prev, 2) 
                        - compute_pressure_z_grad(pressure->p,i,HEIGHT-1,k) 
                        - (NU / (K[idx])) * U->v_z[idx]
                        + (NU) * (compute_velocity_xx_grad(Eta->v_z,i,HEIGHT-1,k)  
                                    + ((Zeta->v_z[down_idx] - 2.0*Zeta->v_z[idx] + (2.0 * bc_velocity(data, x, vel_y_max, vel_z, t_prev, 2) - Zeta->v_z[idx])) * DY_INVERSE_SQUARE) 
                                    + compute_velocity_zz_grad(U->v_z,i,HEIGHT-1,k) );
        }
    }

    //(i, j, k) = (i,j,DEPTH-1)
    for(int j = 1; j < HEIGHT - 1; j++){
        for(int i = 1; i < WIDTH - 1; i++){
            idx = rowmaj_idx(i,j,DEPTH-1);
            back_idx = rowmaj_idx(i,j,DEPTH-2);
            DTYPE x = i * DX;
            DTYPE y = j * DY;
            DTYPE vel_x = x + DX/2;
            DTYPE vel_y = y + DY/2;

            // This will propagate to the solution U
            g_field->g_x[idx] = forcing(data, vel_x, y, z_max, t_half_prev, 0) 
                        - compute_pressure_x_grad(pressure->p,i,j,DEPTH-1) 
                        - (NU / (K[idx])) * U->v_x[idx]
                        + (NU) * (compute_velocity_xx_grad(Eta->v_x,i,j,DEPTH-1)  
                                    + compute_velocity_yy_grad(Zeta->v_x,i,j,DEPTH-1) 
                                    + ((U->v_x[back_idx] - 2.0*U->v_x[idx] + (2.0 * bc_velocity(data, vel_x, y, vel_z_max, t_prev, 0) - U->v_x[idx])) * DZ_INVERSE_SQUARE) );
            
            // This will propagate to the solution U
            g_field->g_y[idx] = forcing(data, x, vel_y, z_max, t_half_prev, 1) 
                        - compute_pressure_y_grad(pressure->p,i,j,DEPTH-1) 
                        - (NU / (K[idx])) * U->v_y[idx]
                        + (NU) * (compute_velocity_xx_grad(Eta->v_y,i,j,DEPTH-1)  
                                    + compute_velocity_yy_grad(Zeta->v_y,i,j,DEPTH-1) 
                                    + ((U->v_y[back_idx] - 2.0*U->v_y[idx] + (2.0 * bc_velocity(data, x, vel_y, vel_z_max, t_prev, 1) - U->v_y[idx])) * DZ_INVERSE_SQUARE) );

        }
    }
}

typedef enum {
    VX = 0,
    VY = 1,
    VZ = 2
} VelocityComponent;

static inline DTYPE upper_xx_grad(const DTYPE *field, size_t idx, size_t left_idx, DTYPE bc_val) {
    return (field[left_idx] - 2.0 * field[idx] + (2.0 * bc_val - field[idx])) * DX_INVERSE_SQUARE;
}

static inline DTYPE upper_yy_grad(const DTYPE *field, size_t idx, size_t down_idx, DTYPE bc_val) {
    return (field[down_idx] - 2.0 * field[idx] + (2.0 * bc_val - field[idx])) * DY_INVERSE_SQUARE;
}

static inline DTYPE upper_zz_grad(const DTYPE *field, size_t idx, size_t back_idx, DTYPE bc_val) {
    return (field[back_idx] - 2.0 * field[idx] + (2.0 * bc_val - field[idx])) * DZ_INVERSE_SQUARE;
}


DTYPE g_value(size_t i, size_t j, size_t k,
              DTYPE *pressure_star,
              DTYPE k_i,
              DTYPE *Eta_prev,
              DTYPE *Zeta_prev,
              DTYPE *U_prev,
              int time_step,
              const Data *data,
              int v_component)
{
    size_t idx = rowmaj_idx(i, j, k);

    DTYPE t_half_prev = (time_step - 0.5) * DT;
    DTYPE t_prev      = (time_step - 1.0) * DT;

    DTYPE x = i * DX;
    DTYPE y = j * DY;
    DTYPE z = k * DZ;

    DTYPE vel_x = x + DX / 2.0;
    DTYPE vel_y = y + DY / 2.0;
    DTYPE vel_z = z + DZ / 2.0;

    DTYPE x_max = (WIDTH  - 1) * DX;
    DTYPE y_max = (HEIGHT - 1) * DY;
    DTYPE z_max = (DEPTH  - 1) * DZ;

    DTYPE vel_x_max = x_max + DX / 2.0;
    DTYPE vel_y_max = y_max + DY / 2.0;
    DTYPE vel_z_max = z_max + DZ / 2.0;

    size_t left_idx = (i > 0) ? rowmaj_idx(i - 1, j, k) : idx;
    size_t down_idx = (j > 0) ? rowmaj_idx(i, j - 1, k) : idx;
    size_t back_idx = (k > 0) ? rowmaj_idx(i, j, k - 1) : idx;

    DTYPE forcing_term, pressure_grad, lap_xx, lap_yy, lap_zz;

    switch (v_component) {

        case VX:
            /*
             * Old compute_g writes g_x only for:
             *   1 <= i <= WIDTH-2
             *   1 <= j <= HEIGHT-1
             *   1 <= k <= DEPTH-1
             */
            if (!(i >= 1 && i <= WIDTH - 2 &&
                  j >= 1 && j <= HEIGHT - 1 &&
                  k >= 1 && k <= DEPTH - 1)) {
                return 0.0;
            }

            forcing_term  = forcing(data, vel_x, y, z, t_half_prev, VX);
            pressure_grad = compute_pressure_x_grad(pressure_star, i, j, k);

            lap_xx = compute_velocity_xx_grad(Eta_prev, i, j, k);

            if (j == HEIGHT - 1) {
                lap_yy = upper_yy_grad(
                    Zeta_prev, idx, down_idx,
                    bc_velocity(data, vel_x, vel_y_max, z, t_prev, VX)
                );
            } else {
                lap_yy = compute_velocity_yy_grad(Zeta_prev, i, j, k);
            }

            if (k == DEPTH - 1) {
                lap_zz = upper_zz_grad(
                    U_prev, idx, back_idx,
                    bc_velocity(data, vel_x, y, vel_z_max, t_prev, VX)
                );
            } else {
                lap_zz = compute_velocity_zz_grad(U_prev, i, j, k);
            }

            return forcing_term
                 - pressure_grad
                 - (NU / k_i) * U_prev[idx]
                 + NU * (lap_xx + lap_yy + lap_zz);

        case VY:
            /*
             * Old compute_g writes g_y only for:
             *   1 <= i <= WIDTH-1
             *   1 <= j <= HEIGHT-2
             *   1 <= k <= DEPTH-1
             */
            if (!(i >= 1 && i <= WIDTH - 1 &&
                  j >= 1 && j <= HEIGHT - 2 &&
                  k >= 1 && k <= DEPTH - 1)) {
                return 0.0;
            }

            forcing_term  = forcing(data, x, vel_y, z, t_half_prev, VY);
            pressure_grad = compute_pressure_y_grad(pressure_star, i, j, k);

            if (i == WIDTH - 1) {
                lap_xx = upper_xx_grad(
                    Eta_prev, idx, left_idx,
                    bc_velocity(data, vel_x_max, vel_y, z, t_prev, VY)
                );
            } else {
                lap_xx = compute_velocity_xx_grad(Eta_prev, i, j, k);
            }

            lap_yy = compute_velocity_yy_grad(Zeta_prev, i, j, k);

            if (k == DEPTH - 1) {
                lap_zz = upper_zz_grad(
                    U_prev, idx, back_idx,
                    bc_velocity(data, x, vel_y, vel_z_max, t_prev, VY)
                );
            } else {
                lap_zz = compute_velocity_zz_grad(U_prev, i, j, k);
            }

            return forcing_term
                 - pressure_grad
                 - (NU / k_i) * U_prev[idx]
                 + NU * (lap_xx + lap_yy + lap_zz);

        case VZ:
            /*
             * Old compute_g writes g_z only for:
             *   1 <= i <= WIDTH-1
             *   1 <= j <= HEIGHT-1
             *   1 <= k <= DEPTH-2
             */
            if (!(i >= 1 && i <= WIDTH - 1 &&
                  j >= 1 && j <= HEIGHT - 1 &&
                  k >= 1 && k <= DEPTH - 2)) {
                return 0.0;
            }

            forcing_term  = forcing(data, x, y, vel_z, t_half_prev, VZ);
            pressure_grad = compute_pressure_z_grad(pressure_star, i, j, k);

            if (i == WIDTH - 1) {
                lap_xx = upper_xx_grad(
                    Eta_prev, idx, left_idx,
                    bc_velocity(data, vel_x_max, y, vel_z, t_prev, VZ)
                );
            } else {
                lap_xx = compute_velocity_xx_grad(Eta_prev, i, j, k);
            }

            if (j == HEIGHT - 1) {
                lap_yy = upper_yy_grad(
                    Zeta_prev, idx, down_idx,
                    bc_velocity(data, x, vel_y_max, vel_z, t_prev, VZ)
                );
            } else {
                lap_yy = compute_velocity_yy_grad(Zeta_prev, i, j, k);
            }

            lap_zz = compute_velocity_zz_grad(U_prev, i, j, k);

            return forcing_term
                 - pressure_grad
                 - (NU / k_i) * U_prev[idx]
                 + NU * (lap_xx + lap_yy + lap_zz);

        default:
            return 0.0;
    }
} 