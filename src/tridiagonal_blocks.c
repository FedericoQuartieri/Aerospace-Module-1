#include "tridiagonal_blocks.h"
#include <stdio.h>


// Thomas algorithm for symmetric tridiagonal matrix:
// Diagonal: (1 - 2*w), Off-diagonals: w (both sub and super)
// where w = -γΔx⁻²
void Thomas_Same_Direction(const DTYPE *__restrict__ w, 
                               unsigned int n,
                               DTYPE *__restrict__ tmp,
                               DTYPE *__restrict__ rhs,
                               DTYPE *__restrict__ u,
                               double v_boundary,
                               DTYPE delta_space 
                            ) 
{
    // Check input 
    if (!w || !tmp || !rhs || !u || n == 0) {
        return; 
    }
    
    // Forward elimination step
    DTYPE norm_coeff;                           
    tmp[0] = 0.0;

    // Set the u[0] left boundary

    rhs[0] = u[0]; // Left boundary value setted before Thomas, in the update_left_boundary 

    //printf("\t\t%f\t\t", u[n-1]);
    for(int i = 1; i < n - 1; i++){
        DTYPE w_i = w[i];

        norm_coeff = 1.0 / ((1.0 - 2.0 * w_i) - w_i * tmp[i - 1]); 

        tmp[i] = w_i * norm_coeff;

        rhs[i] = (rhs[i] - w_i * rhs[i - 1]) * norm_coeff;
    }
    
    // Backward substitution

    // Set the u[n-1] right boundary
    // u[n-1] is already setted (before thomas, in the update_right_boundary)
    // u[n-1] = rhs[n-1]
    
    for(int i = 1; i < n; i++){
        u[n - 1 - i] = rhs[n - 1 - i] - tmp[n - 1 - i] * u[n - i];
    }
}


void Thomas_Different_Direction(const DTYPE *__restrict__ w, 
                               unsigned int n,
                               DTYPE *__restrict__ tmp,
                               DTYPE *__restrict__ rhs,
                               DTYPE *__restrict__ u,
                               double v_boundary,
                               DTYPE delta_space 

                            ) 
{
    // Check input 
    if (!w || !tmp || !rhs || !u || n == 0) {
        return; 
    }
    
    // Forward elimination step
    
    DTYPE norm_coeff;                           
    tmp[0] = 0.0;

    // Set left boundary value
    rhs[0] = u[0]; // Already setted by the update_velocity_function()

    //printf("\t\t%f\t\t", u[n-1]);
    for(int i = 1; i < n-1; i++){
        DTYPE w_i = w[i];

        norm_coeff = 1.0 / ((1.0 - 2.0 * w_i) - w_i * tmp[i - 1]); 

        tmp[i] = w_i * norm_coeff;

        rhs[i] = (rhs[i] - w_i * rhs[i - 1]) * norm_coeff;
    }

    // Backward substitution 
    // for non-tangent components of the right boundary velocity
    rhs[n-1] = rhs[n-1] - 2.0 * w[n-1] * u[n-1]; // rhs = rhs +2*w*U_ex where u[n-1] set to U_ex
    norm_coeff = 1.0 / ((1.0 - 3.0 * w[n-1]) - w[n-1] * tmp[n - 2]);
    rhs[n-1] = (rhs[n-1] - w[n-1]*rhs[n-2]) * norm_coeff;

    u[n - 1] = rhs[n - 1];
    for(int i = 1; i < n; i++){
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
        for(int i = 1; i < n - 1; i++){
            norm_coeff = 1.0 / ((1.0 - 2.0 * w) - w * tmp[i - 1]); 
            tmp[i] = w * norm_coeff;  // Super-diagonal coefficient
            rhs[i] = (rhs[i] - w * rhs[i - 1]) * norm_coeff;  // Sub-diagonal is also w
        }

        norm_coeff = 1.0/((1.0 - 1.0 * w) - w * tmp[n - 2]);
        rhs[n-1] = (rhs[n-1] - w * rhs[n-2]) * norm_coeff;

        // Backward substitution
        u[n - 1] = rhs[n - 1];
        for(int i = 1; i < n; i++){
            u[n - 1 - i] = rhs[n - 1 - i] - tmp[n - 1 - i] * u[n - i];
        }
}

void solve_Dxx_tridiag_blocks(DTYPE *Eta_next_component, DTYPE *rhs, DTYPE *Gamma, function_handle v_boundary, bool same_direction){

    // Initialize temporary arrays 
    DTYPE *w = (DTYPE *) malloc(GRID_SIZE);
    DTYPE *tmp = (DTYPE *) malloc(GRID_SIZE);
    memset(tmp, 0, GRID_SIZE);

    // for(int i=0; i< GRID_SIZE; i++){
    //     w[i] = -Gamma[i] * DX_INVERSE_SQUARE;
    // }

    for(int i = 0; i < WIDTH; i++){
        for(int j = 0; j < HEIGHT; j++){
            for(int k = 0; k < DEPTH; k++){
                size_t idx = rowmaj_idx(i,j,k);
                w[idx] = -Gamma[i] * DX_INVERSE_SQUARE;
            }
        }
    }

    if(same_direction){
        /* Solving for each row of the domain, one at a time. */
        for (int k = 0; k < DEPTH; k++) {
            for (int j = 0; j < HEIGHT; j++) { 
                /* Here we solve for a single block. */
                size_t off = k * (HEIGHT * WIDTH) + j * WIDTH; 

                // Physical coordinates, pass to eval_function() 
                DTYPE y = j*DY + DY/2; DTYPE z = k*DZ + DZ/2; // !! missing time

                Thomas_Same_Direction(w + off, WIDTH, tmp + off, rhs + off, Eta_next_component + off,
                                    eval_function(v_boundary, 0, y, z, 0, 0),
                                    DX);
                //Eta_next_component[0]= dirichlet_left(w + off, rhs + off, Eta_next_component + off)
            }
        }
    } else {
        /* Solving for each row of the domain, one at a time. */
        for (int k = 0; k < DEPTH; k++) {
            for (int j = 0; j < HEIGHT; j++) {
                /* Here we solve for a single block. */
                size_t off = k * (HEIGHT * WIDTH) + j * WIDTH; 

                // Physical coordinates, pass to eval_function() 
                DTYPE y = j*DY + DY/2; DTYPE z = k*DZ + DZ/2; // !! missing time

                Thomas_Different_Direction(w + off, WIDTH, tmp + off, rhs + off, Eta_next_component + off,
                                    eval_function(v_boundary, 0, y, z, 0, 0),
                                    DX);
                //Eta_next_component[0]= dirichlet_left(w + off, f_field_component + off, Eta_next_component + off)
            }
        }
    }
 
    free(w);
    free(tmp);
}

void solve_Dyy_tridiag_blocks(DTYPE *Zeta_next, DTYPE *rhs, DTYPE *Gamma, function_handle v_boundary, bool same_direction){
    // Buffer riutilizzati per ogni colonna (i,k)
    DTYPE *f_block   = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *u_block   = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *w_block   = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *tmp_block = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *rhs_block = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));

    if (!f_block || !u_block || !w_block || !tmp_block|| !rhs_block) {
        free(f_block); free(u_block); free(w_block); free(tmp_block); free(rhs_block);
        return;
    }


    if(same_direction){
        for (int k = 0; k < DEPTH; ++k) {
            for (int i = 0; i < WIDTH; ++i) {
                size_t off = (size_t)k * (HEIGHT * WIDTH) + i; 

                // gather lungo y (stride = WIDTH)
                for (int j = 0; j < HEIGHT; ++j){
                    size_t idx = off + (size_t)j * WIDTH;
                    rhs_block[j] = rhs[idx];
                    w_block[j] = - Gamma[idx] * DY_INVERSE_SQUARE;
                }

                // missed: set u_block boundary value before Thomas !
                u_block[0] = Zeta_next[off]; // left bc: u[0]
                u_block[HEIGHT-1] = Zeta_next[off + (HEIGHT-1)*WIDTH]; // right bc: u[n-1]

                // Physical coordinates, pass to eval_function() 
                DTYPE x = i*DX + DX/2; DTYPE z = k*DZ + DZ/2; // !! missing time

                Thomas_Same_Direction(w_block, HEIGHT, tmp_block, rhs_block, u_block,
                                    eval_function(v_boundary, x, 0, z, 0, 1),
                                    DY);

                // scatter risultato
                for (int j = 0; j < HEIGHT; ++j){
                    size_t idx = off + (size_t)j * WIDTH;
                    Zeta_next[idx] = u_block[j];
                }
            }
        }
    } else {    
        for (int k = 0; k < DEPTH; ++k) {
            for (int i = 0; i < WIDTH; ++i) {
                size_t off = (size_t)k * (HEIGHT * WIDTH) + i; 

                // gather lungo y (stride = WIDTH)
                for (int j = 0; j < HEIGHT; ++j){
                    size_t idx = off + (size_t)j * WIDTH;
                    rhs_block[j] = rhs[idx];
                    w_block[j] = - Gamma[idx] * DY_INVERSE_SQUARE;
                }

                // missed: set u_block boundary value before Thomas !
                u_block[0] = Zeta_next[off]; // left bc: u[0]
                u_block[HEIGHT-1] = Zeta_next[off + (HEIGHT-1)*WIDTH]; // right bc: u[n-1]

                // Physical coordinates, pass to eval_function() 
                DTYPE x = i*DX + DX/2; DTYPE z = k*DZ + DZ/2; // !! missing time

                
                Thomas_Different_Direction(w_block, HEIGHT, tmp_block, rhs_block, u_block,
                                    eval_function(v_boundary, x, 0, z, 0, 1),
                                    DY);

                // scatter risultato
                for (int j = 0; j < HEIGHT; ++j){
                    size_t idx = off + (size_t)j * WIDTH;
                    Zeta_next[idx] = u_block[j];
                }
            }
        }
    }

    free(tmp_block);
    free(w_block);
    free(u_block);
    free(f_block);
    free(rhs_block);
}

void solve_Dzz_tridiag_blocks(DTYPE *U_next, DTYPE *rhs, DTYPE *Gamma, function_handle v_boundary, bool same_direction){
    // Buffer riutilizzati per ogni colonna (i,k)
    DTYPE *f_block   = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *u_block   = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *w_block   = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *tmp_block = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *rhs_block = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));

    if (!f_block || !u_block || !w_block || !tmp_block || !rhs_block) {
        free(f_block); free(u_block); free(w_block); free(tmp_block); free(rhs_block);
        return;
    }

    if(same_direction){
        for (int j = 0; j < HEIGHT; ++j) {
            for (int i = 0; i < WIDTH; ++i) {
                size_t off = (size_t)j * WIDTH + i;

                // gather lungo z (stride = HEIGHT * WIDTH)
                for (int k = 0; k < DEPTH; ++k){
                    size_t idx = off + (size_t)k * (HEIGHT * WIDTH);
                    rhs_block[k] = rhs[idx];
                    w_block[k] = - Gamma[idx] * DZ_INVERSE_SQUARE;
                }

                // missed: set u_block boundary value before Thomas !
                u_block[0] = U_next[off]; // Left bc: u[0]
                u_block[DEPTH-1] = U_next[off + (DEPTH-1)*HEIGHT*WIDTH]; // Right bc: u[n-1]

                // Physical coordinates, pass to eval_function() 
                DTYPE x = i*DX + DX/2; DTYPE y = j*DY + DY/2; // !! missing time

                
                Thomas_Same_Direction(w_block, DEPTH, tmp_block, rhs_block, u_block,
                                    eval_function(v_boundary, x, y, 0, 0, 2),
                                    DZ);

                // scatter risultato
                for (int k = 0; k < DEPTH; ++k){
                    size_t idx = off + (size_t)k * (HEIGHT * WIDTH);
                    U_next[idx] = u_block[k];
                }
            }
        }
    } else {
        for (int j = 0; j < HEIGHT; ++j) {
            for (int i = 0; i < WIDTH; ++i) {
                size_t off = (size_t)j * WIDTH + i;

                // gather lungo z (stride = HEIGHT * WIDTH)
                for (int k = 0; k < DEPTH; ++k){
                    size_t idx = off + (size_t)k * (HEIGHT * WIDTH);
                    rhs_block[k] = rhs[idx];
                    w_block[k] = - Gamma[idx] * DZ_INVERSE_SQUARE;
                }

                // missed: set u_block boundary value before Thomas !
                u_block[0] = U_next[off]; // Left bc: u[0]
                u_block[DEPTH-1] = U_next[off + (DEPTH-1)*HEIGHT*WIDTH]; // Right bc: u[n-1]

                // Physical coordinates, pass to eval_function() 
                DTYPE x = i*DX + DX/2; DTYPE y = j*DY + DY/2; // !! missing time

                
                Thomas_Different_Direction(w_block, DEPTH, tmp_block, rhs_block, u_block,
                                    eval_function(v_boundary, x, y, 0, 0, 2),
                                    DZ);

                // scatter risultato
                for (int k = 0; k < DEPTH; ++k){
                    size_t idx = off + (size_t)k * (HEIGHT * WIDTH);
                    U_next[idx] = u_block[k];
                }
            }
        }
    }

    free(tmp_block);
    free(w_block);
    free(u_block);
    free(f_block);
    free(rhs_block);
}