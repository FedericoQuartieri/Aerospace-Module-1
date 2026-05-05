#include "pressure_system.h"
#include "velocity_field.h"
#include <stdio.h> // test only 

void solve_pressure_system(VelocityField U_next, Pressure *pressure, Pressure *pressure_star, 
                            Pressure *psi, Pressure *phi_lower, Pressure *phi_higher, uint64_t *solver_time) {
                 
    uint64_t time = time_ns();
    compute_Psi(U_next, psi); // Dxx
    solver_time[3] += (time_ns() - time) / 1e6;

    time = time_ns();
    //compute_Phi_lower(psi, phi_lower); // Dyy
    optimize_compute_Phi_lower(psi,  phi_lower); // Dyy
    solver_time[4] += (time_ns() - time) / 1e6;

    time = time_ns();
    //compute_Phi_higher(phi_lower, phi_higher); // Dzz
    optimize_compute_Phi_higher(phi_lower, phi_higher); // Dzz
    solver_time[5] += (time_ns() - time) / 1e6;

    time = time_ns();
    compute_pressure(phi_higher, pressure, pressure_star);
    solver_time[6] += (time_ns() - time) / 1e6;

}

void compute_Psi(VelocityField U_next, Pressure *psi){
    // Initialize temporary arrays 
    DTYPE *tmp = (DTYPE *) malloc(WIDTH * sizeof(DTYPE));
    DTYPE *rhs = (DTYPE *) malloc(GRID_SIZE);

    /* 
        In the left boundaries points impose the div(U) = 0
    */
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                // Left boundaries divergence is 0.0
                if(k==0 || j==0 || i ==0) {
                    rhs[idx] = 0.0;
                    continue;
                }

                rhs[idx] = (compute_velocity_x_grad(U_next.v_x, i, j, k) +
                              compute_velocity_y_grad(U_next.v_y, i, j, k) +
                              compute_velocity_z_grad(U_next.v_z, i, j, k)) *  (-1.0 / DT);

            }
        }
    }

    DTYPE w = - DX_INVERSE_SQUARE;

    //printf("\nPressure system, checking last value of rhs (gradient of boundaries index): \n "
      //  ": %f \n", rhs[rowmaj_idx(WIDTH-1,HEIGHT-1, DEPTH-1)]);

    //printf("\nPressure system, checking last value of rhs (gradient of boundaries index): \n "
       // ": %f \n", rhs[rowmaj_idx(3,3,3)]);

    /* Solving for each row of the domain, one at a time. */
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            /* Here we solve for a single block. */
            size_t off = k * (HEIGHT * WIDTH) + j * WIDTH;

            Thomas_Pressure(w, WIDTH, tmp, rhs + off, psi->p + off);
        }
    }

    free(tmp);
    free(rhs);
}

void compute_Phi_lower(Pressure *psi, Pressure *phi_lower){
    // Initialize temporary arrays 
    DTYPE *rhs_block = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *u_block = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *tmp = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));

    
    Pressure rhs;
    initialize_pressure(&rhs);
        
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                rhs.p[idx] = psi->p[idx];   
            }
        }
    }  

    DTYPE W_Y = - DY_INVERSE_SQUARE;

    // Loop sui sistemi 1D lungo Y (su ogni colonna i,k)
    for (int k = 0; k < DEPTH; ++k) {
        for (int i = 0; i < WIDTH; ++i) {
            
            size_t off = (size_t)k * (HEIGHT * WIDTH) + i; // Offset per la colonna (i,k)

             // 1. GATHER (Raccogli i dati lungo Y)
             for (int j = 0; j < HEIGHT; ++j){
                size_t idx = off + (size_t)j * WIDTH; // Indice 3D
                
                // Assumendo il passo di PRESSIONE (no Gamma)
                rhs_block[j] = rhs.p[idx]; 
            }   

            Thomas_Pressure(W_Y, HEIGHT, tmp, rhs_block, u_block);

            // 3. SCATTER (Spargi il risultato in phi_lower)
            for (int j = 0; j < HEIGHT; ++j){
                size_t idx = off + (size_t)j * WIDTH; // Indice 3D
                
                phi_lower->p[idx] = u_block[j]; // Scrivi nell'array 3D di output
            }
        }
    }

    free(tmp);
    free(rhs_block);
    free(u_block);
    free_pressure(&rhs);
};
                        
void compute_Phi_higher(Pressure *phi_lower, Pressure *phi_higher){
    DTYPE *rhs_block = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *u_block = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *tmp_thomas = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    
    Pressure rhs;
        initialize_pressure(&rhs);
        for(int k = 0; k < DEPTH; k++){
            for(int j = 0; j < HEIGHT; j++){
                for(int i = 0; i < WIDTH; i++){
                    size_t idx = rowmaj_idx(i,j,k);

                    rhs.p[idx] = phi_lower->p[idx];
                    
                }
            }
        }

    DTYPE w = - DZ_INVERSE_SQUARE;

    // Loop sui sistemi 1D lungo Y (su ogni colonna i,k)
    for (int j = 0; j < HEIGHT; ++j) {
        for (int i = 0; i < WIDTH; ++i) {
            
            size_t off = (size_t)j * WIDTH + i;

            // 1. GATHER (Raccogli i dati lungo Y)
            for (int k = 0; k < DEPTH; ++k){
                size_t idx = off + (size_t)k * (HEIGHT * WIDTH); // Indice 3D
                
                // Assumendo il passo di PRESSIONE (no Gamma)
                rhs_block[k] = rhs.p[idx]; 
            }

            Thomas_Pressure(w, DEPTH, tmp_thomas, rhs_block, u_block);

            // 3. SCATTER (Spargi il risultato in phi_lower)
            for (int k = 0; k < DEPTH; ++k){
                size_t idx = off + (size_t)k * (HEIGHT * WIDTH); // Indice 3D
                
                phi_higher->p[idx] = u_block[k]; // Scrivi nell'array 3D di output
            }
        }
    }

    free(tmp_thomas);
    free(rhs_block);
    free(u_block);
    free_pressure(&rhs);
};

/* Solving for Dzz */
void optimize_compute_Phi_higher(Pressure *phi_lower, Pressure *phi_higher){

    /*
        The coefficient Gamma is constant, rhs = phi_lower. Moving on the z-columns
        Using the sliding window technique to compute the tridiagonal system.
        phi_lower->p is overwritten by Thomas as rhs workspace.
    */

    int slice_dim = 16; // Number of SIMD vectors treated as one cache-friendly block.
    int slice_size = slice_dim * VLEN;

    DTYPE *simd_tmp = (DTYPE *) malloc(DEPTH * sizeof(DTYPE) * slice_size);
    DTYPE *scalar_rhs = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *rhs = phi_lower->p;

    if(!simd_tmp || !scalar_rhs){
        free(simd_tmp);
        free(scalar_rhs);
        return;
    }

    VTYPE vect_one = VSET1((DTYPE) 1.0);
    VTYPE vect_two = VSET1((DTYPE) 2.0);

    /* Same for all, w is constant. */
    DTYPE w_z = (DTYPE) -DZ_INVERSE_SQUARE;
    VTYPE W_Z = VSET1((DTYPE) -DZ_INVERSE_SQUARE);
    VTYPE norm_coeff_left = VDIV(vect_one, VSUB(vect_one, VMUL(vect_two, W_Z)));
    VTYPE tmp_left = VMUL(VMUL(W_Z, norm_coeff_left), vect_two);

    for(int j = 0; j < HEIGHT; j++){
        int i = 0;

        for(; i + slice_size <= WIDTH; i += slice_size) {
            size_t off = (size_t)j * WIDTH + i;

            /* Left Neumann row: tmp[0] = 2*w / (1 - 2*w), rhs[0] /= (1 - 2*w). */
            for(int s = 0; s < slice_dim; s++){
                size_t field_slice_idx = off + (size_t)s * VLEN;
                int slice_idx = s * VLEN;

                VTYPE rhs_i = VLOAD(&rhs[field_slice_idx]);
                rhs_i = VMUL(rhs_i, norm_coeff_left);
                VSTORE(&rhs[field_slice_idx], rhs_i);
                VSTORE(&simd_tmp[slice_idx], tmp_left);
            }

            /* Forward pass. */
            for(int k = 1; k < DEPTH - 1; k++){
                size_t idx = off + (size_t)k * (WIDTH * HEIGHT);
                size_t prev_idx = off + (size_t)(k - 1) * (WIDTH * HEIGHT);

                for(int s = 0; s < slice_dim; s++){
                    size_t field_slice_idx = idx + (size_t)s * VLEN;
                    size_t prev_field_slice_idx = prev_idx + (size_t)s * VLEN;
                    int slice_idx = k * slice_size + s * VLEN;
                    int prev_slice_idx = (k - 1) * slice_size + s * VLEN;

                    VTYPE vect_tmp_i_prev = VLOAD(&simd_tmp[prev_slice_idx]);
                    VTYPE vect_norm_coeff = VSUB(VSUB(vect_one, VMUL(vect_two, W_Z)), VMUL(W_Z, vect_tmp_i_prev));
                    vect_norm_coeff = VDIV(vect_one, vect_norm_coeff);

                    VTYPE vect_tmp_i = VMUL(W_Z, vect_norm_coeff);
                    VSTORE(&simd_tmp[slice_idx], vect_tmp_i);

                    VTYPE vect_rhs = VLOAD(&rhs[field_slice_idx]);
                    VTYPE vect_rhs_prev = VLOAD(&rhs[prev_field_slice_idx]);
                    vect_rhs = VMUL(VSUB(vect_rhs, VMUL(W_Z, vect_rhs_prev)), vect_norm_coeff);
                    VSTORE(&rhs[field_slice_idx], vect_rhs);
                }
            }

            /* Right Neumann row. */
            for(int s = 0; s < slice_dim; s++){
                size_t field_slice_idx = off + (size_t)(DEPTH - 1) * (WIDTH * HEIGHT) + (size_t)s * VLEN;
                size_t prev_field_slice_idx = off + (size_t)(DEPTH - 2) * (WIDTH * HEIGHT) + (size_t)s * VLEN;
                int prev_slice_idx = (DEPTH - 2) * slice_size + s * VLEN;

                VTYPE tmp_right_prev = VLOAD(&simd_tmp[prev_slice_idx]);
                VTYPE norm_coeff_right = VSUB(VSUB(vect_one, W_Z), VMUL(W_Z, tmp_right_prev));
                norm_coeff_right = VDIV(vect_one, norm_coeff_right);

                VTYPE rhs_i = VLOAD(&rhs[field_slice_idx]);
                VTYPE rhs_prev = VLOAD(&rhs[prev_field_slice_idx]);
                rhs_i = VMUL(VSUB(rhs_i, VMUL(W_Z, rhs_prev)), norm_coeff_right);

                VSTORE(&phi_higher->p[field_slice_idx], rhs_i);
            }

            /* Backward pass. */
            for(int k = DEPTH - 2; k >= 0; k--){
                size_t idx = off + (size_t)k * (WIDTH * HEIGHT);

                for(int s = 0; s < slice_dim; s++){
                    size_t field_slice_idx = idx + (size_t)s * VLEN;
                    int slice_idx = k * slice_size + s * VLEN;

                    VTYPE rhs_i = VLOAD(&rhs[field_slice_idx]);
                    VTYPE tmp_i = VLOAD(&simd_tmp[slice_idx]);
                    VTYPE u_ip1 = VLOAD(&phi_higher->p[idx + (WIDTH * HEIGHT) + (size_t)s * VLEN]);
                    VTYPE u_i = VSUB(rhs_i, VMUL(tmp_i, u_ip1));

                    VSTORE(&phi_higher->p[field_slice_idx], u_i);
                }
            }
        }

        /* Fallback in case WIDTH is not divisible by slice_size. */
        for(; i < WIDTH; i++){
            size_t off = (size_t)j * WIDTH + i;

            for(int k = 0; k <DEPTH; k++){
                scalar_rhs[k] = rhs[off + (size_t)k * (WIDTH * HEIGHT)];
            }

            DTYPE norm_coeff = 1.0 / (1.0 - 2.0 * w_z);
            simd_tmp[0] = (2.0 * w_z) * norm_coeff;
            scalar_rhs[0] *= norm_coeff;

            for(int k = 1; k < DEPTH - 1; k++){
                norm_coeff = 1.0 / ((1.0 - 2.0 * w_z) - w_z * simd_tmp[k - 1]);
                simd_tmp[k] = w_z * norm_coeff;
                scalar_rhs[k] = (scalar_rhs[k] - w_z * scalar_rhs[k - 1]) * norm_coeff;
            }

            norm_coeff = 1.0 / ((1.0 - w_z) - w_z * simd_tmp[DEPTH - 2]);
            scalar_rhs[DEPTH - 1] = (scalar_rhs[DEPTH - 1] - w_z * scalar_rhs[DEPTH - 2]) * norm_coeff;

            for(int k = 0; k < DEPTH; k++){
                size_t idx = off + (size_t)k * (WIDTH * HEIGHT);
                rhs[idx] = scalar_rhs[k];
            }

            phi_higher->p[off + (size_t)(DEPTH - 1) * (WIDTH * HEIGHT)] = scalar_rhs[DEPTH - 1];
            for(int k = DEPTH - 2; k >= 0; k--){
                size_t idx = off + (size_t)k * (WIDTH * HEIGHT);
                size_t next_idx = idx + (WIDTH * HEIGHT);

                phi_higher->p[idx] = scalar_rhs[k] - simd_tmp[k] * phi_higher->p[next_idx];
            }
        }
    }

    free(simd_tmp);
    free(scalar_rhs);
}

/* Solver for Dyy */
void optimize_compute_Phi_lower(Pressure *psi, Pressure *phi_lower){
    
    int slice_dim = 16; // Number of SIMD vectors treated as one cache-friendly block.
    int slice_size = slice_dim * VLEN;

    DTYPE *simd_tmp = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE) * slice_size);
    DTYPE *scalar_rhs = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *rhs = psi->p;

    if(!simd_tmp || !scalar_rhs){
        free(simd_tmp);
        free(scalar_rhs);
        return;
    }

    VTYPE vect_one = VSET1((DTYPE) 1.0);
    VTYPE vect_two = VSET1((DTYPE) 2.0);

    /* Same for all, w is constant. */
    DTYPE w_y = (DTYPE) -DY_INVERSE_SQUARE;
    VTYPE W_Y = VSET1((DTYPE) -DY_INVERSE_SQUARE);
    VTYPE norm_coeff_left = VDIV(vect_one, VSUB(vect_one, VMUL(vect_two, W_Y)));
    VTYPE tmp_left = VMUL(VMUL(W_Y, norm_coeff_left), vect_two);

    for(int k = 0; k < DEPTH; k++){
        int i = 0;

        for(; i + slice_size <= WIDTH; i += slice_size) {
            size_t off = (size_t)k * (HEIGHT * WIDTH) + i;

            /* Left Neumann row: tmp[0] = 2*w / (1 - 2*w), rhs[0] /= (1 - 2*w). */
            for(int s = 0; s < slice_dim; s++){
                size_t field_slice_idx = off + (size_t)s * VLEN;
                int slice_idx = s * VLEN;

                VTYPE rhs_i = VLOAD(&rhs[field_slice_idx]);
                rhs_i = VMUL(rhs_i, norm_coeff_left);
                VSTORE(&rhs[field_slice_idx], rhs_i);
                VSTORE(&simd_tmp[slice_idx], tmp_left);
            }

            /* Forward pass. */
            for(int j = 1; j < HEIGHT - 1; j++){
                size_t idx = off + (size_t)j * WIDTH;
                size_t prev_idx = off + (size_t)(j - 1) * WIDTH;

                for(int s = 0; s < slice_dim; s++){
                    size_t field_slice_idx = idx + (size_t)s * VLEN;
                    size_t prev_field_slice_idx = prev_idx + (size_t)s * VLEN;
                    int slice_idx = j * slice_size + s * VLEN;
                    int prev_slice_idx = (j - 1) * slice_size + s * VLEN;

                    VTYPE vect_tmp_i_prev = VLOAD(&simd_tmp[prev_slice_idx]);
                    VTYPE vect_norm_coeff = VSUB(VSUB(vect_one, VMUL(vect_two, W_Y)), VMUL(W_Y, vect_tmp_i_prev));
                    vect_norm_coeff = VDIV(vect_one, vect_norm_coeff);

                    VTYPE vect_tmp_i = VMUL(W_Y, vect_norm_coeff);
                    VSTORE(&simd_tmp[slice_idx], vect_tmp_i);

                    VTYPE vect_rhs = VLOAD(&rhs[field_slice_idx]);
                    VTYPE vect_rhs_prev = VLOAD(&rhs[prev_field_slice_idx]);
                    vect_rhs = VMUL(VSUB(vect_rhs, VMUL(W_Y, vect_rhs_prev)), vect_norm_coeff);
                    VSTORE(&rhs[field_slice_idx], vect_rhs);
                }
            }

            /* Right Neumann row. */
            for(int s = 0; s < slice_dim; s++){
                size_t field_slice_idx = off + (size_t)(HEIGHT - 1) * WIDTH + (size_t)s * VLEN;
                size_t prev_field_slice_idx = off + (size_t)(HEIGHT - 2) * WIDTH + (size_t)s * VLEN;
                int prev_slice_idx = (HEIGHT - 2) * slice_size + s * VLEN;

                VTYPE tmp_right_prev = VLOAD(&simd_tmp[prev_slice_idx]);
                VTYPE norm_coeff_right = VSUB(VSUB(vect_one, W_Y), VMUL(W_Y, tmp_right_prev));
                norm_coeff_right = VDIV(vect_one, norm_coeff_right);

                VTYPE rhs_i = VLOAD(&rhs[field_slice_idx]);
                VTYPE rhs_prev = VLOAD(&rhs[prev_field_slice_idx]);
                rhs_i = VMUL(VSUB(rhs_i, VMUL(W_Y, rhs_prev)), norm_coeff_right);

                VSTORE(&rhs[field_slice_idx], rhs_i);
                VSTORE(&phi_lower->p[field_slice_idx], rhs_i);
            }

            /* Backward pass. */
            for(int j = HEIGHT - 2; j >= 0; j--){
                size_t idx = off + (size_t)j * WIDTH;

                for(int s = 0; s < slice_dim; s++){
                    size_t field_slice_idx = idx + (size_t)s * VLEN;
                    int slice_idx = j * slice_size + s * VLEN;

                    VTYPE rhs_i = VLOAD(&rhs[field_slice_idx]);
                    VTYPE tmp_i = VLOAD(&simd_tmp[slice_idx]);
                    VTYPE u_ip1 = VLOAD(&phi_lower->p[idx + WIDTH + (size_t)s * VLEN]);
                    VTYPE u_i = VSUB(rhs_i, VMUL(tmp_i, u_ip1));

                    VSTORE(&phi_lower->p[field_slice_idx], u_i);
                }
            }
        }

        /* Fallback in case WIDTH is not divisible by slice_size. */
        for(; i < WIDTH; i++){
            size_t off = (size_t)k * (HEIGHT * WIDTH) + i;

            for(int j = 0; j < HEIGHT; j++){
                scalar_rhs[j] = rhs[off + (size_t)j * WIDTH];
            }

            DTYPE norm_coeff = 1.0 / (1.0 - 2.0 * w_y);
            simd_tmp[0] = (2.0 * w_y) * norm_coeff;
            scalar_rhs[0] *= norm_coeff;

            for(int j = 1; j < HEIGHT - 1; j++){
                norm_coeff = 1.0 / ((1.0 - 2.0 * w_y) - w_y * simd_tmp[j - 1]);
                simd_tmp[j] = w_y * norm_coeff;
                scalar_rhs[j] = (scalar_rhs[j] - w_y * scalar_rhs[j - 1]) * norm_coeff;
            }

            norm_coeff = 1.0 / ((1.0 - w_y) - w_y * simd_tmp[HEIGHT - 2]);
            scalar_rhs[HEIGHT - 1] = (scalar_rhs[HEIGHT - 1] - w_y * scalar_rhs[HEIGHT - 2]) * norm_coeff;

            for(int j = 0; j < HEIGHT; j++){
                size_t idx = off + (size_t)j * WIDTH;
                rhs[idx] = scalar_rhs[j];
            }

            phi_lower->p[off + (size_t)(HEIGHT - 1) * WIDTH] = scalar_rhs[HEIGHT - 1];
            for(int j = HEIGHT - 2; j >= 0; j--){
                size_t idx = off + (size_t)j * WIDTH;
                size_t next_idx = idx + WIDTH;

                phi_lower->p[idx] = scalar_rhs[j] - simd_tmp[j] * phi_lower->p[next_idx];
            }
        }
    }

    free(simd_tmp);
    free(scalar_rhs);
}

void compute_pressure(Pressure *phi_higher, Pressure *pressure, Pressure *pressure_star){
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);
                pressure->p[idx] += phi_higher->p[idx];

                // Now we compute p_star, needed for the next timestep n+1=N
                // The current timestep is n and this pressure is p(n + 1/2) = p(n - 1/2) + phi(n + 1/2)
                // Now we consider the next timestep n + 1 = N
                // To compute g for the timestep N we need p_star(N + 1/2)
                // By definition p_star(N + 1/2) = p(N - 1/2) + phi(N - 1/2)
                // now we substitute N = n + 1 and we get that:
                // p_star(N + 1/2) = p(n + 1/2) + phi(n + 1/2)
                // where p(n + 1/2) = p(n - 1/2) + phi(n + 1/2)
                pressure_star->p[idx] = pressure->p[idx] + phi_higher->p[idx]; 
            }
        }
    }
};
