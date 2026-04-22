#include "tridiagonal_blocks.h"
#include <stdio.h>

void simd_Thomas_Algorithm(DTYPE *__restrict__ simd_w, unsigned int n, DTYPE *__restrict__ simd_tmp, DTYPE *__restrict__ simd_rhs, DTYPE *__restrict__ simd_u, bool same_direction){

    if(!simd_w || !simd_tmp || !simd_rhs || !simd_u){
        return;
    }

    /* Set to zero the tmp[0] for each element of the vector */
    VTYPE zero = VSET1((DTYPE) 0.0);
    VSTORE(&simd_tmp[0], zero);

    /* rhs[0] = u[0] */
    VTYPE vect_u0 = VLOAD(&simd_u[0]);
    VSTORE(&simd_rhs[0], vect_u0);

    /* Forward pass */

    VTYPE vect_one = VSET1((DTYPE) 1.0);
    VTYPE vect_two = VSET1((DTYPE) 2.0);

    for(unsigned int i = 1; i < n-1; i++){
        VTYPE vect_w_i = VLOAD(&simd_w[i*VLEN]);
        VTYPE vect_tmp_i_prev = VLOAD(&simd_tmp[(i-1)*VLEN]);
        
        VTYPE vect_norm_coeff = VSUB(VSUB(vect_one, VMUL(vect_two, vect_w_i)), VMUL(vect_w_i, vect_tmp_i_prev));
        vect_norm_coeff = VDIV(vect_one, vect_norm_coeff);

        VTYPE vect_tmp_i = VMUL(vect_w_i, vect_norm_coeff);
        VSTORE(&simd_tmp[i*VLEN], vect_tmp_i);
 
        VTYPE vect_rhs_i = VLOAD(&simd_rhs[i*VLEN]);
        VTYPE vect_rhs_i_prev = VLOAD(&simd_rhs[(i-1)*VLEN]);

        vect_rhs_i = VMUL(VSUB(vect_rhs_i, VMUL(vect_w_i, vect_rhs_i_prev)), vect_norm_coeff);

        VSTORE(&simd_rhs[i*VLEN], vect_rhs_i);
    }

    VTYPE vect_three = VSET1((DTYPE)3.0);

    if (!same_direction) {
        VTYPE w_last   = VLOAD(&simd_w[(n-1)*VLEN]);
        VTYPE rhs_last = VLOAD(&simd_rhs[(n-1)*VLEN]);
        VTYPE rhs_prev = VLOAD(&simd_rhs[(n-2)*VLEN]);
        VTYPE tmp_prev = VLOAD(&simd_tmp[(n-2)*VLEN]);
        VTYPE u_last   = VLOAD(&simd_u[(n-1)*VLEN]);

        rhs_last = VSUB(rhs_last, VMUL(VMUL(vect_two, w_last), u_last));

        VTYPE denom = VSUB(VSUB(vect_one, VMUL(vect_three, w_last)),
                        VMUL(w_last, tmp_prev));
        VTYPE norm  = VDIV(vect_one, denom);

        rhs_last = VMUL(VSUB(rhs_last, VMUL(w_last, rhs_prev)), norm);

        VSTORE(&simd_rhs[(n-1)*VLEN], rhs_last);
        VSTORE(&simd_u[(n-1)*VLEN], rhs_last);
    }

    /* Backword pass */
    for (int i = (int)n - 2; i >= 0; --i) {
        VTYPE rhs_i = VLOAD(&simd_rhs[i * VLEN]);
        VTYPE tmp_i = VLOAD(&simd_tmp[i * VLEN]);
        VTYPE u_ip1 = VLOAD(&simd_u[(i + 1) * VLEN]);

        VTYPE u_i = VSUB(rhs_i, VMUL(tmp_i, u_ip1));
        VSTORE(&simd_u[i * VLEN], u_i);
    }
}

void vectorized_solve_Dyy_tridiag_blocks(DTYPE *__restrict__ Zeta_next, DTYPE *__restrict__ rhs, DTYPE *__restrict__ Gamma, const Data *__restrict__ data, bool same_direction, int v_component, int time_step) {
    /* 
        Using SIMD instructions to vectorized the tridiagonal system
        this is based on the VLEN of the SIMD instructions
    */
    DTYPE *simd_u_block = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE) * VLEN);
    DTYPE *simd_w_block = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE) * VLEN);
    DTYPE *simd_rhs_block = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE) * VLEN);
    DTYPE *simd_tmp_block = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE) * VLEN);

    VTYPE minus_dy = VSET1((DTYPE) -DY_INVERSE_SQUARE);

    DTYPE t = time_step * DT;
    DTYPE bvx, bvy, bvz;

    if (!simd_u_block || !simd_w_block || !simd_rhs_block || !simd_tmp_block) {
        free(simd_u_block);
        free(simd_w_block);
        free(simd_rhs_block);
        free(simd_tmp_block);
        return;
    }

    /* 
        Loop over the domain but jumping of VLEN row
        Warning: check if the size is divisible by VLEN
    */
    for(int k = 0; k < DEPTH; k ++){
        int i = 0;

        for(; i + VLEN <= WIDTH; i += VLEN){
            size_t off = (size_t)k * (WIDTH*HEIGHT) + i;

            /* Gather the weights */
            for(int j = 0; j < HEIGHT; j++){
                size_t idx = off + (size_t)j * WIDTH;
                
                /* Load instructions into vector register */
                VTYPE vect_w = VLOAD(&Gamma[idx]);
                vect_w = VMUL(vect_w, minus_dy);
                VSTORE(&simd_w_block[j*VLEN], vect_w);

                /* rhs has multiple row stored contiguosly, so we use j*VLEN */
                VTYPE vect_rhs = VLOAD(&rhs[idx]);
                VSTORE(&simd_rhs_block[j*VLEN], vect_rhs); 

            }

            for (int x = 0; x < VLEN; x++) {
                DTYPE bvx0, bvy0, bvz0;
                DTYPE bvx1, bvy1, bvz1;

                get_boundary_velocity(i + x, 0,        k, t, data, &bvx0, &bvy0, &bvz0);
                get_boundary_velocity(i + x, HEIGHT-1, k, t, data, &bvx1, &bvy1, &bvz1);

                simd_u_block[x] = 
                    (v_component == 0) ? bvx0 :
                    (v_component == 1) ? bvy0 : bvz0;

                simd_u_block[(HEIGHT - 1) * VLEN + x] =
                    (v_component == 0) ? bvx1 :
                    (v_component == 1) ? bvy1 : bvz1;
            }

            /* Thomas_simd will receive a set of columns now */
            simd_Thomas_Algorithm(simd_w_block, HEIGHT, simd_tmp_block, simd_rhs_block, simd_u_block, same_direction);

            /* Scatter the result */
            for(int j = 0; j < HEIGHT; j++){
                size_t idx = off + (size_t)j * WIDTH;

                VTYPE vect_u = VLOAD(&simd_u_block[j*VLEN]);
                VSTORE(&Zeta_next[idx], vect_u);
            }
            
        }

        /* Fallback in case WIDTH%VLEN != 0 */
        for(; i < WIDTH; i++){
            size_t off = (size_t)k * (HEIGHT * WIDTH) + i;

            for (int j = 0; j < HEIGHT; ++j) {
                size_t idx = off + (size_t)j * WIDTH;
                simd_rhs_block[j] = rhs[idx];
                simd_w_block[j] = -Gamma[idx] * DY_INVERSE_SQUARE;
            }

            get_boundary_velocity(i, 0, k, t, data, &bvx, &bvy, &bvz);
            simd_u_block[0] = (v_component == 0) ? bvx : (v_component == 1) ? bvy : bvz;
            get_boundary_velocity(i, HEIGHT - 1, k, t, data, &bvx, &bvy, &bvz);
            simd_u_block[HEIGHT - 1] = (v_component == 0) ? bvx : (v_component == 1) ? bvy : bvz;

            Thomas_Algorithm(simd_w_block, HEIGHT, simd_tmp_block, simd_rhs_block, simd_u_block, same_direction);

            for (int j = 0; j < HEIGHT; ++j) {
                size_t idx = off + (size_t)j * WIDTH;
                Zeta_next[idx] = simd_u_block[j];
            }
        }
    }

    free(simd_u_block);
    free(simd_rhs_block);
    free(simd_tmp_block);
    free(simd_w_block);
}

void vectorized_solve_Dzz_tridiag_blocks(DTYPE *__restrict__ U_next, DTYPE *__restrict__ rhs, DTYPE *__restrict__ Gamma, const Data *__restrict__ data, bool same_direction, int v_component, int time_step) {
    /* 
        Using SIMD instructions to vectorized the tridiagonal system
        this is based on the VLEN of the SIMD instructions
    */
    DTYPE *simd_u_block = (DTYPE *) malloc(DEPTH * sizeof(DTYPE) * VLEN);
    DTYPE *simd_w_block = (DTYPE *) malloc(DEPTH * sizeof(DTYPE) * VLEN);
    DTYPE *simd_rhs_block = (DTYPE *) malloc(DEPTH * sizeof(DTYPE) * VLEN);
    DTYPE *simd_tmp_block = (DTYPE *) malloc(DEPTH * sizeof(DTYPE) * VLEN);

    VTYPE minus_dz = VSET1((DTYPE) - DZ_INVERSE_SQUARE);

    DTYPE t = time_step * DT;
    DTYPE bvx, bvy, bvz;

    if (!simd_u_block || !simd_w_block || !simd_rhs_block || !simd_tmp_block) {
        free(simd_u_block);
        free(simd_w_block);
        free(simd_rhs_block);
        free(simd_tmp_block);
        return;
    }

    /* 
        Loop over the domain but jumping of VLEN row
        Warning: check if the size is divisible by VLEN
    */
    for(int j = 0; j < HEIGHT; j++){
        int i = 0;

        for(; i + VLEN <= WIDTH; i += VLEN){
            size_t off = (size_t)j * WIDTH + i;

            /* Gather the weights */
            for(int k = 0; k < DEPTH; k++){
                size_t idx = off + (size_t)k * (WIDTH * HEIGHT);
                
                /* Load instructions into vector register */
                VTYPE vect_w = VLOAD(&Gamma[idx]);
                vect_w = VMUL(vect_w, minus_dz);
                VSTORE(&simd_w_block[k*VLEN], vect_w);

                /* rhs has multiple row stored contiguosly, so we use j*VLEN */
                VTYPE vect_rhs = VLOAD(&rhs[idx]);
                VSTORE(&simd_rhs_block[k*VLEN], vect_rhs); 

            }

            for (int x = 0; x < VLEN; x++) {
                DTYPE bvx0, bvy0, bvz0;
                DTYPE bvx1, bvy1, bvz1;

                get_boundary_velocity(i + x, j,        0, t, data, &bvx0, &bvy0, &bvz0);
                get_boundary_velocity(i + x, j, DEPTH-1, t, data, &bvx1, &bvy1, &bvz1);

                simd_u_block[x] = 
                    (v_component == 0) ? bvx0 :
                    (v_component == 1) ? bvy0 : bvz0;

                simd_u_block[(DEPTH - 1) * VLEN + x] =
                    (v_component == 0) ? bvx1 :
                    (v_component == 1) ? bvy1 : bvz1;
            }

            /* Thomas_simd will receive a set of columns now */
            simd_Thomas_Algorithm(simd_w_block, DEPTH, simd_tmp_block, simd_rhs_block, simd_u_block, same_direction);

            /* Scatter the result */
            for(int k = 0; k < DEPTH; k++){
                size_t idx = off + (size_t)k * (WIDTH * HEIGHT);

                VTYPE vect_u = VLOAD(&simd_u_block[k*VLEN]);
                VSTORE(&U_next[idx], vect_u);
            }
            
        }

        /* Fallback in case WIDTH%VLEN != 0 */
        for(; i < WIDTH; i++){
            size_t off = (size_t)j * WIDTH + i;

            for (int k = 0; k < DEPTH; ++k) {
                size_t idx = off + (size_t)k * (WIDTH * HEIGHT);
                simd_rhs_block[k] = rhs[idx];
                simd_w_block[k] = -Gamma[idx] * DZ_INVERSE_SQUARE;
            }

            get_boundary_velocity(i, j, 0, t, data, &bvx, &bvy, &bvz);
            simd_u_block[0] = (v_component == 0) ? bvx : (v_component == 1) ? bvy : bvz;
            get_boundary_velocity(i, j, DEPTH - 1, t, data, &bvx, &bvy, &bvz);
            simd_u_block[DEPTH - 1] = (v_component == 0) ? bvx : (v_component == 1) ? bvy : bvz;

            Thomas_Algorithm(simd_w_block, DEPTH, simd_tmp_block, simd_rhs_block, simd_u_block, same_direction);

            for (int k = 0; k < DEPTH; ++k) {
                size_t idx = off + (size_t)k * (WIDTH * HEIGHT);
                U_next[idx] = simd_u_block[k];
            }
        }
    }

    free(simd_u_block);
    free(simd_rhs_block);
    free(simd_tmp_block);
    free(simd_w_block);
}

// Thomas algorithm for symmetric tridiagonal matrix:
// Diagonal: (1 - 2*w), Off-diagonals: w (both sub and super)
// where w = -γΔx⁻²
void Thomas_Algorithm(const DTYPE *__restrict__ w, 
                               unsigned int n,
                               DTYPE *__restrict__ tmp,
                               DTYPE *__restrict__ rhs,
                               DTYPE *__restrict__ u,
                               bool same_direction
                            ) 
{ 
    if (!w || !tmp || !rhs || !u || n == 0) {
        return; 
    }
    
    // Forward elimination step
    DTYPE norm_coeff;                           
    tmp[0] = 0.0;


    rhs[0] = u[0]; // Left boundary value setted before Thomas, in the update_left_boundary 

    //printf("\t\t%f\t\t", u[n-1]);
    for(unsigned int i = 1; i < n - 1; i++){
        DTYPE w_i = w[i];

        norm_coeff = 1.0 / ((1.0 - 2.0 * w_i) - w_i * tmp[i - 1]); 

        tmp[i] = w_i * norm_coeff;

        rhs[i] = (rhs[i] - w_i * rhs[i - 1]) * norm_coeff;
    }
    
    // Backward substitution

    // Set the u[n-1] right boundary
    // u[n-1] is already setted (before thomas, in the update_right_boundary)
    // u[n-1] = rhs[n-1]

    if(same_direction){
        u[n-1] = u[n-1]; // already setted by the update_right_boundary
    } else {
        rhs[n-1] = rhs[n-1] - 2.0 * w[n-1] * u[n-1]; // rhs = rhs +2*w*U_ex where u[n-1] set to U_ex
        norm_coeff = 1.0 / ((1.0 - 3.0 * w[n-1]) - w[n-1] * tmp[n - 2]);
        rhs[n-1] = (rhs[n-1] - w[n-1]*rhs[n-2]) * norm_coeff;

        u[n - 1] = rhs[n - 1];

        /*
        // GIULIO RIGHT BOUNDARIES Backward substitution with w = -γΔx⁻²
        // for non-tangent components of the right boundary velocity
        rhs[n-1] = rhs[n-1] - (2.0/3.0) * w[n-1] * u[n-1]; // rhs = rhs + (-2/3w) * U_ex where u[n-1] set to U_ex
        norm_coeff = 1.0 / ((1.0 - w[n-1]) - ((w[n-1]/3.0) * tmp[n - 2])); // -gamma/3delta^2 * U_n-1 + (gamma/delta^2 + 1)*U_n = rhs + (2gamma/3delta^2) * U_ex
        rhs[n-1] = (rhs[n-1] - w[n-1]*rhs[n-2]) * norm_coeff;
        */
    }

    for(unsigned int i = 1; i < n; i++){
        u[n - 1 - i] = rhs[n - 1 - i] - tmp[n - 1 - i] * u[n - i];
    }
}

/* Warning: w is constant = - DX_INVERSE_SQUARE , so it should be constant and not an array */
void Thomas_Pressure(const DTYPE  w, 
                    unsigned int  n,
                    DTYPE *__restrict__ tmp,
                    DTYPE *__restrict__ rhs,
                    DTYPE *__restrict__ u) 
{
    // Check input 
        if (!tmp || !rhs || !u || n == 0) {
            return; 
        }

        // Thomas algorithm for symmetric tridiagonal matrix:
        // Diagonal: (1 - 2*w), Off-diagonals: w (both sub and super)
        // This matches the discretization: (1 + 2γΔx⁻²) with off-diagonals -γΔx⁻²
        // where w = -γΔx⁻²

        //first equation is: (1-2w_0)p0 + (w_-1 + w1)p1 = f0

        // !! We are imposing homogeneous Neumann condition !!

        // Forward elimination step
        DTYPE norm_coeff = 1.0 / (1.0 - 2.0 * w);                           
        tmp[0] = (2.0 * w) * norm_coeff;  // Super-diagonal coefficient
        rhs[0] = rhs[0] * norm_coeff;
        for(unsigned int i = 1; i < n - 1; i++){
            norm_coeff = 1.0 / ((1.0 - 2.0 * w) - w * tmp[i - 1]); 
            tmp[i] = w * norm_coeff;  // Super-diagonal coefficient
            rhs[i] = (rhs[i] - w * rhs[i - 1]) * norm_coeff;  // Sub-diagonal is also w
        }

        norm_coeff = 1.0/((1.0 - 1.0 * w) - w * tmp[n - 2]);
        rhs[n-1] = (rhs[n-1] - w * rhs[n-2]) * norm_coeff;

        // Backward substitution
        u[n - 1] = rhs[n - 1];
        for(unsigned int i = 1; i < n; i++){
            u[n - 1 - i] = rhs[n - 1 - i] - tmp[n - 1 - i] * u[n - i];
        }
}

void solve_Dxx_tridiag_blocks(DTYPE *__restrict__ Eta_next_component, DTYPE *__restrict__ rhs, DTYPE *__restrict__ Gamma, const Data *data, bool same_direction, int v_component, int time_step){

    // Initialize temporary arrays 
    DTYPE *w = (DTYPE *) malloc(GRID_SIZE);
    DTYPE *tmp = (DTYPE *) malloc(GRID_SIZE);
    memset(tmp, 0, GRID_SIZE);
    DTYPE t = time_step * DT;
    DTYPE bvx, bvy, bvz;

    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);
                w[idx] = -Gamma[idx] * DX_INVERSE_SQUARE;
            }
        }
    }

    
    /* Solving for each row of the domain, one at a time. */
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) { 
            /* Here we solve for a single block. */
            size_t off = k * (HEIGHT * WIDTH) + j * WIDTH; 
                
            /* Compute the left and right boundaries */
            get_boundary_velocity(0, j, k, t, data, &bvx, &bvy, &bvz);
            Eta_next_component[off] = (v_component == 0) ? bvx : (v_component == 1) ? bvy : bvz;
            get_boundary_velocity(WIDTH-1, j, k, t, data, &bvx, &bvy, &bvz);
            Eta_next_component[off + WIDTH-1] = (v_component == 0) ? bvx : (v_component == 1) ? bvy : bvz;

            Thomas_Algorithm(w + off, WIDTH, tmp + off, rhs + off, Eta_next_component + off, same_direction);
        }
    }
 
    free(w);
    free(tmp);
}

void solve_Dyy_tridiag_blocks(DTYPE *Zeta_next, DTYPE *rhs, DTYPE *Gamma, const Data *data, bool same_direction, int v_component, int time_step){
    // Buffer riutilizzati per ogni colonna (i,k)
    DTYPE *f_block   = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *u_block   = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *w_block   = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *tmp_block = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *rhs_block = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));

    DTYPE t = time_step * DT;
    DTYPE bvx, bvy, bvz;

    if (!f_block || !u_block || !w_block || !tmp_block|| !rhs_block) {
        free(f_block); free(u_block); free(w_block); free(tmp_block); free(rhs_block);
        return;
    }

    for (int k = 0; k < DEPTH; ++k) {
        for (int i = 0; i < WIDTH; ++i) {
            size_t off = (size_t)k * (HEIGHT * WIDTH) + i; 

            // gather lungo y (stride = WIDTH)
            for (int j = 0; j < HEIGHT; ++j){
                size_t idx = off + (size_t)j * WIDTH;
                rhs_block[j] = rhs[idx];
                w_block[j] = - Gamma[idx] * DY_INVERSE_SQUARE;
            }

            /* Compute the left and right boundaries */
            get_boundary_velocity(i, 0 , k, t, data, &bvx, &bvy, &bvz);
            u_block[0] = (v_component == 0) ? bvx : (v_component == 1) ? bvy : bvz;
            get_boundary_velocity(i, HEIGHT-1, k, t, data, &bvx, &bvy, &bvz);
            u_block[HEIGHT-1] = (v_component == 0) ? bvx : (v_component == 1) ? bvy : bvz;

            Thomas_Algorithm(w_block, HEIGHT, tmp_block, rhs_block, u_block, same_direction);

            // scatter risultato
            for (int j = 0; j < HEIGHT; ++j){
                size_t idx = off + (size_t)j * WIDTH;
                Zeta_next[idx] = u_block[j];
            }
        }
    }  

    free(tmp_block);
    free(w_block);
    free(u_block);
    free(f_block);
    free(rhs_block);
}

void solve_Dzz_tridiag_blocks(DTYPE *U_next, DTYPE *rhs, DTYPE *Gamma, const Data *data, bool same_direction, int v_component, int time_step){
    // Buffer riutilizzati per ogni colonna (i,k)
    DTYPE *f_block   = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *u_block   = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *w_block   = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *tmp_block = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *rhs_block = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));

    DTYPE t = time_step * DT;
    DTYPE bvx, bvy, bvz;

    if (!f_block || !u_block || !w_block || !tmp_block || !rhs_block) {
        free(f_block); free(u_block); free(w_block); free(tmp_block); free(rhs_block);
        return;
    }

    for (int j = 0; j < HEIGHT; ++j) {
        for (int i = 0; i < WIDTH; ++i) {
            size_t off = (size_t)j * WIDTH + i;

            // gather lungo z (stride = HEIGHT * WIDTH)
            for (int k = 0; k < DEPTH; ++k){
                size_t idx = off + (size_t)k * (HEIGHT * WIDTH);
                rhs_block[k] = rhs[idx];
                w_block[k] = - Gamma[idx] * DZ_INVERSE_SQUARE;
            }

            /* Compute the left and right boundaries */
            get_boundary_velocity(i, j, 0, t, data, &bvx, &bvy, &bvz);
            u_block[0] = (v_component == 0) ? bvx : (v_component == 1) ? bvy : bvz;
            get_boundary_velocity(i, j, DEPTH-1, t, data, &bvx, &bvy, &bvz);
            u_block[DEPTH-1] = (v_component == 0) ? bvx : (v_component == 1) ? bvy : bvz;
    
            Thomas_Algorithm(w_block, DEPTH, tmp_block, rhs_block, u_block, same_direction);

            // scatter risultato
            for (int k = 0; k < DEPTH; ++k){
                size_t idx = off + (size_t)k * (HEIGHT * WIDTH);
                U_next[idx] = u_block[k];
            }            
        }
    }

    free(tmp_block);
    free(w_block);
    free(u_block);
    free(f_block);
    free(rhs_block);
}
