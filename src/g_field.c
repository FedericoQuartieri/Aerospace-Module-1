#include "g_field.h"


void initialize_g_field(GField *g_field){
    g_field->g_x = (DTYPE *) malloc(GRID_SIZE);
    g_field->g_y = (DTYPE *) malloc(GRID_SIZE);
    g_field->g_z = (DTYPE *) malloc(GRID_SIZE);

    memset(g_field->g_x, 0, GRID_SIZE);
    memset(g_field->g_y, 0, GRID_SIZE);
    memset(g_field->g_z, 0, GRID_SIZE);
}

void compute_g(GField *g_field, function forcing, Pressure *pressure, DTYPE *K, VelocityField *Eta, VelocityField *Zeta, VelocityField *U, int time_step, function v_boundary){


    size_t idx, left_idx, down_idx, back_idx;
    // TODO: check boundaries condition
    for(int k = 1; k < DEPTH-1; k++){
        for(int j = 1; j < HEIGHT-1; j++){
            for(int i = 1; i < WIDTH-1; i++){
                
                idx = rowmaj_idx(i,j,k);

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




    // (i,j,k) = (WIDTH-1,HEIGHT-1,DEPTH-1)
    idx = rowmaj_idx(WIDTH-1,HEIGHT-1,DEPTH-1);
    left_idx = rowmaj_idx(WIDTH-2,HEIGHT-1,DEPTH-1);
    down_idx = rowmaj_idx(WIDTH-1,HEIGHT-2,DEPTH-1);
    back_idx = rowmaj_idx(WIDTH-1,HEIGHT-1,DEPTH-2);
    ((Eta->v_x[left_idx] - 2*Eta->v_x[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,DEPTH-1, time_step, 0) - Eta->v_x[idx])) * DX_INVERSE_SQUARE);
    g_field->g_x[idx] = forcing((WIDTH-1)*DX, (HEIGHT-1)*DY, (DEPTH-1)*DZ, time_step*DT, 0) 
                    - compute_pressure_x_grad(pressure->p,WIDTH-1,HEIGHT-1,DEPTH-1) 
                    - (NU / (2.0 * K[idx])) * U->v_x[idx]
                    + (NU/2.0) * (((Eta->v_x[left_idx] - 2.0*Eta->v_x[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,DEPTH-1, time_step, 0) - Eta->v_x[idx])) * DX_INVERSE_SQUARE)  
                                + ((Zeta->v_x[down_idx] - 2.0*Zeta->v_x[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,DEPTH-1, time_step, 0) - Zeta->v_x[idx])) * DY_INVERSE_SQUARE) 
                                + ((U->v_x[back_idx] - 2.0*U->v_x[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,DEPTH-1, time_step, 0) - U->v_x[idx])) * DZ_INVERSE_SQUARE) );

    g_field->g_y[idx] = forcing((WIDTH-1)*DX, (HEIGHT-1)*DY, (DEPTH-1)*DZ, time_step*DT, 1) 
                    - compute_pressure_y_grad(pressure->p,WIDTH-1,HEIGHT-1,DEPTH-1) 
                    - (NU / (2.0 * K[idx])) * U->v_y[idx]
                    + (NU/2.0) * (((Eta->v_y[left_idx] - 2.0*Eta->v_y[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,DEPTH-1, time_step, 1) - Eta->v_y[idx])) * DX_INVERSE_SQUARE)  
                                + ((Zeta->v_y[down_idx] - 2.0*Zeta->v_y[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,DEPTH-1, time_step, 1) - Zeta->v_y[idx])) * DY_INVERSE_SQUARE) 
                                + ((U->v_y[back_idx] - 2.0*U->v_y[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,DEPTH-1, time_step, 1) - U->v_y[idx])) * DZ_INVERSE_SQUARE) );

    g_field->g_z[idx] = forcing((WIDTH-1)*DX, (HEIGHT-1)*DY, (DEPTH-1)*DZ, time_step*DT, 2) 
                        - compute_pressure_z_grad(pressure->p,WIDTH-1,HEIGHT-1,DEPTH-1) 
                        - (NU / (2.0 * K[idx])) * U->v_z[idx]
                    + (NU/2.0) * (((Eta->v_z[left_idx] - 2.0*Eta->v_z[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,DEPTH-1, time_step, 2) - Eta->v_z[idx])) * DX_INVERSE_SQUARE)  
                                + ((Zeta->v_z[down_idx] - 2.0*Zeta->v_z[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,DEPTH-1, time_step, 2) - Zeta->v_z[idx])) * DY_INVERSE_SQUARE) 
                                + ((U->v_z[back_idx] - 2.0*U->v_z[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,DEPTH-1, time_step, 2) - U->v_z[idx])) * DZ_INVERSE_SQUARE) );

    // (i,j,k) = (WIDTH-1,HEIGHT-1,k)
    for(int k = 1; k < DEPTH - 1; k++){
        idx = rowmaj_idx(WIDTH-1,HEIGHT-1,k);
        left_idx = rowmaj_idx(WIDTH-2,HEIGHT-1,k);
        down_idx = rowmaj_idx(WIDTH-1,HEIGHT-2,k);

        g_field->g_x[idx] = forcing((WIDTH-1)*DX, (HEIGHT-1)*DY, k*DZ, time_step*DT, 0) 
                    - compute_pressure_x_grad(pressure->p,WIDTH-1,HEIGHT-1,k) 
                    - (NU / (2.0 * K[idx])) * U->v_x[idx]
                    + (NU/2.0) * (((Eta->v_x[left_idx] - 2.0*Eta->v_x[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,k, time_step, 0) - Eta->v_x[idx])) * DX_INVERSE_SQUARE)  
                                + ((Zeta->v_x[down_idx] - 2.0*Zeta->v_x[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,k, time_step, 0) - Zeta->v_x[idx])) * DY_INVERSE_SQUARE) 
                                + compute_velocity_zz_grad(U->v_x,WIDTH-1,HEIGHT-1,k) );
        
        g_field->g_y[idx] = forcing((WIDTH-1)*DX, (HEIGHT-1)*DY, k*DZ, time_step*DT, 1) 
                    - compute_pressure_y_grad(pressure->p,WIDTH-1,HEIGHT-1,k) 
                    - (NU / (2.0 * K[idx])) * U->v_y[idx]
                    + (NU/2.0) * (((Eta->v_y[left_idx] - 2.0*Eta->v_y[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,k, time_step, 1) - Eta->v_y[idx])) * DX_INVERSE_SQUARE)  
                                + ((Zeta->v_y[down_idx] - 2.0*Zeta->v_y[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,k, time_step, 1) - Zeta->v_y[idx])) * DY_INVERSE_SQUARE) 
                                + compute_velocity_zz_grad(U->v_y,WIDTH-1,HEIGHT-1,k) );
        
        g_field->g_z[idx] = forcing((WIDTH-1)*DX, (HEIGHT-1)*DY, k*DZ, time_step*DT, 2) 
                    - compute_pressure_z_grad(pressure->p,WIDTH-1,HEIGHT-1,k) 
                    - (NU / (2.0 * K[idx])) * U->v_z[idx]
                    + (NU/2.0) * (((Eta->v_z[left_idx] - 2.0*Eta->v_z[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,k, time_step, 2) - Eta->v_z[idx])) * DX_INVERSE_SQUARE)  
                                + ((Zeta->v_z[down_idx] - 2.0*Zeta->v_z[idx] + (2.0 * v_boundary(WIDTH-1,HEIGHT-1,k, time_step, 2) - Zeta->v_z[idx])) * DY_INVERSE_SQUARE) 
                                + compute_velocity_zz_grad(U->v_z,WIDTH-1,HEIGHT-1,k) ); 
    }   

    // (i,j,k) = (WIDTH-1,j,DEPTH-1)
    for(int j = 1; j < HEIGHT - 1; j++){
        idx = rowmaj_idx(WIDTH-1,j,DEPTH-1);
        left_idx = rowmaj_idx(WIDTH-2,j,DEPTH-1);
        back_idx = rowmaj_idx(WIDTH-1,j,DEPTH-2);   
        g_field->g_x[idx] = forcing((WIDTH-1)*DX, j*DY, (DEPTH-1)*DZ, time_step*DT, 0) 
                    - compute_pressure_x_grad(pressure->p,WIDTH-1,j,DEPTH-1) 
                    - (NU / (2.0 * K[idx])) * U->v_x[idx]
                    + (NU/2.0) * (((Eta->v_x[left_idx] - 2.0*Eta->v_x[idx] + (2.0 * v_boundary(WIDTH-1,j,DEPTH-1, time_step, 0) - Eta->v_x[idx])) * DX_INVERSE_SQUARE)  
                                + compute_velocity_yy_grad(Zeta->v_x,WIDTH-1,j,DEPTH-1) 
                                + ((U->v_x[back_idx] - 2.0*U->v_x[idx] + (2.0 * v_boundary(WIDTH-1,j,DEPTH-1, time_step, 0) - U->v_x[idx])) * DZ_INVERSE_SQUARE) );
        g_field->g_y[idx] = forcing((WIDTH-1)*DX, j*DY, (DEPTH-1)*DZ, time_step*DT, 1) 
                    - compute_pressure_y_grad(pressure->p,WIDTH-1,j,DEPTH-1) 
                    - (NU / (2.0 * K[idx])) * U->v_y[idx]
                    + (NU/2.0) * (((Eta->v_y[left_idx] - 2.0*Eta->v_y[idx] + (2.0 * v_boundary(WIDTH-1,j,DEPTH-1, time_step, 1) - Eta->v_y[idx])) * DX_INVERSE_SQUARE)  
                                + compute_velocity_yy_grad(Zeta->v_y,WIDTH-1,j,DEPTH-1) 
                                + ((U->v_y[back_idx] - 2.0*U->v_y[idx] + (2.0 * v_boundary(WIDTH-1,j,DEPTH-1, time_step, 1) - U->v_y[idx])) * DZ_INVERSE_SQUARE) );
        g_field->g_z[idx] = forcing((WIDTH-1)*DX, j*DY, (DEPTH-1)*DZ, time_step*DT, 2) 
                    - compute_pressure_z_grad(pressure->p,WIDTH-1,j,DEPTH-1) 
                    - (NU / (2.0 * K[idx])) * U->v_z[idx]
                    + (NU/2.0) * (((Eta->v_z[left_idx] - 2.0*Eta->v_z[idx] + (2.0 * v_boundary(WIDTH-1,j,DEPTH-1, time_step, 2) - Eta->v_z[idx])) * DX_INVERSE_SQUARE)  
                                + compute_velocity_yy_grad(Zeta->v_z,WIDTH-1,j,DEPTH-1) 
                                + ((U->v_z[back_idx] - 2.0*U->v_z[idx] + (2.0 * v_boundary(WIDTH-1,j,DEPTH-1, time_step, 2) - U->v_z[idx])) * DZ_INVERSE_SQUARE) );
    }

    // (i,j,k) = (i,HEIGHT-1,DEPTH-1)
    for(int i = 1; i < WIDTH - 1; i++){
        idx = rowmaj_idx(i,HEIGHT-1,DEPTH-1);
        down_idx = rowmaj_idx(i,HEIGHT-2,DEPTH-1);
        back_idx = rowmaj_idx(i,HEIGHT-1,DEPTH-2);   
        g_field->g_x[idx] = forcing(i*DX, (HEIGHT-1)*DY, (DEPTH-1)*DZ, time_step*DT, 0) 
                    - compute_pressure_x_grad(pressure->p,i,HEIGHT-1,DEPTH-1) 
                    - (NU / (2.0 * K[idx])) * U->v_x[idx]
                    + (NU/2.0) * (compute_velocity_xx_grad(Eta->v_x,i,HEIGHT-1,DEPTH-1)  
                                + ((Zeta->v_x[down_idx] - 2.0*Zeta->v_x[idx] + (2.0 * v_boundary(i,HEIGHT-1,DEPTH-1, time_step, 0) - Zeta->v_x[idx])) * DY_INVERSE_SQUARE) 
                                + ((U->v_x[back_idx] - 2.0*U->v_x[idx] + (2.0 * v_boundary(i,HEIGHT-1,DEPTH-1, time_step, 0) - U->v_x[idx])) * DZ_INVERSE_SQUARE) );
        g_field->g_y[idx] = forcing(i*DX, (HEIGHT-1)*DY, (DEPTH-1)*DZ, time_step*DT, 1) 
                    - compute_pressure_y_grad(pressure->p,i,HEIGHT-1,DEPTH-1) 
                    - (NU / (2.0 * K[idx])) * U->v_y[idx]
                    + (NU/2.0) * (compute_velocity_xx_grad(Eta->v_y,i,HEIGHT-1,DEPTH-1)  
                                + ((Zeta->v_y[down_idx] - 2.0*Zeta->v_y[idx] + (2.0 * v_boundary(i,HEIGHT-1,DEPTH-1, time_step, 1) - Zeta->v_y[idx])) * DY_INVERSE_SQUARE) 
                                + ((U->v_y[back_idx] - 2.0*U->v_y[idx] + (2.0 * v_boundary(i,HEIGHT-1,DEPTH-1, time_step, 1) - U->v_y[idx])) * DZ_INVERSE_SQUARE) );
        g_field->g_z[idx] = forcing(i*DX, (HEIGHT-1)*DY, (DEPTH-1)*DZ, time_step*DT, 2) 
                    - compute_pressure_z_grad(pressure->p,i,HEIGHT-1,DEPTH-1) 
                    - (NU / (2.0 * K[idx])) * U->v_z[idx]
                    + (NU/2.0) * (compute_velocity_xx_grad(Eta->v_z,i,HEIGHT-1,DEPTH-1)  
                                + ((Zeta->v_z[down_idx] - 2.0*Zeta->v_z[idx] + (2.0 * v_boundary(i,HEIGHT-1,DEPTH-1, time_step, 2) - Zeta->v_z[idx])) * DY_INVERSE_SQUARE) 
                                + ((U->v_z[back_idx] - 2.0*U->v_z[idx] + (2.0 * v_boundary(i,HEIGHT-1,DEPTH-1, time_step, 2) - U->v_z[idx])) * DZ_INVERSE_SQUARE) );
    }

    // (i, j, k) = (WIDTH-1,j,k)
    for(int k = 1; k < DEPTH - 1; k++){
        for(int j = 1; j < HEIGHT - 1; j++){
            idx = rowmaj_idx(WIDTH-1,j,k);
            left_idx = rowmaj_idx(WIDTH-2,j,k);

            g_field->g_x[idx] = forcing((WIDTH-1)*DX, j*DY, k*DZ, time_step*DT, 0) 
                        - compute_pressure_x_grad(pressure->p,WIDTH-1,j,k) 
                        - (NU / (2.0 * K[idx])) * U->v_x[idx]
                        + (NU/2.0) * (((Eta->v_x[left_idx] - 2.0*Eta->v_x[idx] + (2.0 * v_boundary(WIDTH-1,j,k, time_step, 0) - Eta->v_x[idx])) * DX_INVERSE_SQUARE)  
                                    + compute_velocity_yy_grad(Zeta->v_x,WIDTH-1,j,k) 
                                    + compute_velocity_zz_grad(U->v_x,WIDTH-1,j,k) );
            
            g_field->g_y[idx] = forcing((WIDTH-1)*DX, j*DY, k*DZ, time_step*DT, 1) 
                        - compute_pressure_y_grad(pressure->p,WIDTH-1,j,k) 
                        - (NU / (2.0 * K[idx])) * U->v_y[idx]
                        + (NU/2.0) * (((Eta->v_y[left_idx] - 2.0*Eta->v_y[idx] + (2.0 * v_boundary(WIDTH-1,j,k, time_step, 1) - Eta->v_y[idx])) * DX_INVERSE_SQUARE)  
                                    + compute_velocity_yy_grad(Zeta->v_y,WIDTH-1,j,k) 
                                    + compute_velocity_zz_grad(U->v_y,WIDTH-1,j,k) );
            
            g_field->g_z[idx] = forcing((WIDTH-1)*DX, j*DY, k*DZ, time_step*DT, 2) 
                        - compute_pressure_z_grad(pressure->p,WIDTH-1,j,k) 
                        - (NU / (2.0 * K[idx])) * U->v_z[idx]
                        + (NU/2.0) * (((Eta->v_z[left_idx] - 2.0*Eta->v_z[idx] + (2.0 * v_boundary(WIDTH-1,j,k, time_step, 2) - Eta->v_z[idx])) * DX_INVERSE_SQUARE)  
                                    + compute_velocity_yy_grad(Zeta->v_z,WIDTH-1,j,k) 
                                    + compute_velocity_zz_grad(U->v_z,WIDTH-1,j,k) );
        }
    }

    //(i, j, k) = (i,HEIGHT-1,k)
    for(int k = 1; k < DEPTH - 1; k++){
        for(int i = 1; i < WIDTH - 1; i++){
            idx = rowmaj_idx(i,HEIGHT-1,k);
            down_idx = rowmaj_idx(i,HEIGHT-2,k);

            g_field->g_x[idx] = forcing(i*DX, (HEIGHT-1)*DY, k*DZ, time_step*DT, 0) 
                        - compute_pressure_x_grad(pressure->p,i,HEIGHT-1,k) 
                        - (NU / (2.0 * K[idx])) * U->v_x[idx]
                        + (NU/2.0) * (compute_velocity_xx_grad(Eta->v_x,i,HEIGHT-1,k)  
                                    + ((Zeta->v_x[down_idx] - 2.0*Zeta->v_x[idx] + (2.0 * v_boundary(i,HEIGHT-1,k, time_step, 0) - Zeta->v_x[idx])) * DY_INVERSE_SQUARE) 
                                    + compute_velocity_zz_grad(U->v_x,i,HEIGHT-1,k) );
            
            g_field->g_y[idx] = forcing(i*DX, (HEIGHT-1)*DY, k*DZ, time_step*DT, 1) 
                        - compute_pressure_y_grad(pressure->p,i,HEIGHT-1,k) 
                        - (NU / (2.0 * K[idx])) * U->v_y[idx]
                        + (NU/2.0) * (compute_velocity_xx_grad(Eta->v_y,i,HEIGHT-1,k)  
                                    + ((Zeta->v_y[down_idx] - 2.0*Zeta->v_y[idx] + (2.0 * v_boundary(i,HEIGHT-1,k, time_step, 1) - Zeta->v_y[idx])) * DY_INVERSE_SQUARE) 
                                    + compute_velocity_zz_grad(U->v_y,i,HEIGHT-1,k) );
            
            g_field->g_z[idx] = forcing(i*DX, (HEIGHT-1)*DY, k*DZ, time_step*DT, 2) 
                        - compute_pressure_z_grad(pressure->p,i,HEIGHT-1,k) 
                        - (NU / (2.0 * K[idx])) * U->v_z[idx]
                        + (NU/2.0) * (compute_velocity_xx_grad(Eta->v_z,i,HEIGHT-1,k)  
                                    + ((Zeta->v_z[down_idx] - 2.0*Zeta->v_z[idx] + (2.0 * v_boundary(i,HEIGHT-1,k, time_step, 2) - Zeta->v_z[idx])) * DY_INVERSE_SQUARE) 
                                    + compute_velocity_zz_grad(U->v_z,i,HEIGHT-1,k) );
        }
    }

    //(i, j, k) = (i,j,DEPTH-1)
    for(int j = 1; j < HEIGHT - 1; j++){
        for(int i = 1; i < WIDTH - 1; i++){
            idx = rowmaj_idx(i,j,DEPTH-1);
            back_idx = rowmaj_idx(i,j,DEPTH-2);         
            g_field->g_x[idx] = forcing(i*DX, j*DY, (DEPTH-1)*DZ, time_step*DT, 0) 
                        - compute_pressure_x_grad(pressure->p,i,j,DEPTH-1) 
                        - (NU / (2.0 * K[idx])) * U->v_x[idx]
                        + (NU/2.0) * (compute_velocity_xx_grad(Eta->v_x,i,j,DEPTH-1)  
                                    + compute_velocity_yy_grad(Zeta->v_x,i,j,DEPTH-1) 
                                    + ((U->v_x[back_idx] - 2.0*U->v_x[idx] + (2.0 * v_boundary(i,j,DEPTH-1, time_step, 0) - U->v_x[idx])) * DZ_INVERSE_SQUARE) );
            g_field->g_y[idx] = forcing(i*DX, j*DY, (DEPTH-1)*DZ, time_step*DT, 1) 
                        - compute_pressure_y_grad(pressure->p,i,j,DEPTH-1) 
                        - (NU / (2.0 * K[idx])) * U->v_y[idx]
                        + (NU/2.0) * (compute_velocity_xx_grad(Eta->v_y,i,j,DEPTH-1)  
                                    + compute_velocity_yy_grad(Zeta->v_y,i,j,DEPTH-1) 
                                    + ((U->v_y[back_idx] - 2.0*U->v_y[idx] + (2.0 * v_boundary(i,j,DEPTH-1, time_step, 1) - U->v_y[idx])) * DZ_INVERSE_SQUARE) );
            g_field->g_z[idx] = forcing(i*DX, j*DY, (DEPTH-1)*DZ, time_step*DT, 2) 
                        - compute_pressure_z_grad(pressure->p,i,j,DEPTH-1) 
                        - (NU / (2.0 * K[idx])) * U->v_z[idx]
                        + (NU/2.0) * (compute_velocity_xx_grad(Eta->v_z,i,j,DEPTH-1)  
                                    + compute_velocity_yy_grad(Zeta->v_z,i,j,DEPTH-1) 
                                    + ((U->v_z[back_idx] - 2.0*U->v_z[idx] + (2.0 * v_boundary(i,j,DEPTH-1, time_step, 2) - U->v_z[idx])) * DZ_INVERSE_SQUARE) );
        }
    }


















}

void free_g_field(GField *g_field){
    free(g_field->g_x);
    free(g_field->g_y);
    free(g_field->g_z);
}








