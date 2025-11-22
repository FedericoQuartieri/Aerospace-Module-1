#include "g_field.h"


void initialize_g_field(GField *g_field){
    g_field->g_x = (DTYPE *) malloc(GRID_SIZE);
    g_field->g_y = (DTYPE *) malloc(GRID_SIZE);
    g_field->g_z = (DTYPE *) malloc(GRID_SIZE);

    memset(g_field->g_x, 0, GRID_SIZE);
    memset(g_field->g_y, 0, GRID_SIZE);
    memset(g_field->g_z, 0, GRID_SIZE);
}

void compute_g(GField *g_field, forcing_function_t forcing, Pressure *pressure, DTYPE *K, VelocityField *Eta, VelocityField *Zeta, VelocityField *U, int time_step){

    // TODO: check boundaries condition
    for(int k = 1; k < DEPTH; k++){
        for(int j = 1; j < HEIGHT; j++){
            for(int i = 1; i < WIDTH; i++){
                
                size_t idx = rowmaj_idx(i,j,k);

                g_field->g_x[idx] = forcing(i*DX,j*DY,k*DZ, time_step*DT , 0)
                                    - compute_pressure_x_grad(pressure->p,i,j,k) 
                                    - (NU / (2.0 * K[idx])) * U->v_x[idx]
                                    + (NU/2.0) * (compute_velocity_xx_grad(Eta->v_x,i,j,k)  
                                                + compute_velocity_yy_grad(Zeta->v_x,i,j,k) 
                                                + compute_velocity_zz_grad(U->v_x,i,j,k));
                                    

                g_field->g_y[idx] =  forcing(i*DX,j*DY,k*DZ, time_step*DT , 1)
                                    - compute_pressure_y_grad(pressure->p,i,j,k) 
                                    - (NU / (2.0 * K[idx])) * U->v_y[idx]
                                    + (NU/2.0) * (compute_velocity_xx_grad(Eta->v_y,i,j,k)  
                                                + compute_velocity_yy_grad(Zeta->v_y,i,j,k) 
                                                + compute_velocity_zz_grad(U->v_y,i,j,k)); 
                 

                g_field->g_z[idx] =  forcing(i*DX,j*DY,k*DZ, time_step*DT , 2)
                                    - compute_pressure_z_grad(pressure->p,i,j,k) 
                                    - (NU / (2.0 * K[idx])) * U->v_z[idx]
                                    + (NU/2.0) * (compute_velocity_xx_grad(Eta->v_z,i,j,k)  
                                                + compute_velocity_yy_grad(Zeta->v_z,i,j,k) 
                                                + compute_velocity_zz_grad(U->v_z,i,j,k));
            }
        }
    }
}

void free_g_field(GField *g_field){
    free(g_field->g_x);
    free(g_field->g_y);
    free(g_field->g_z);
}








