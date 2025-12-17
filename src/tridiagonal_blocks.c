#include "tridiagonal_blocks.h"


void Thomas_Same_Direction(const DTYPE *__restrict__ w, 
                               unsigned int n,
                               DTYPE *__restrict__ tmp,
                               DTYPE *__restrict__ rhs,
                               DTYPE *__restrict__ u,
                               DTYPE delta_space 
                            ) 
{
    // Check input 
    if (!w || !tmp || !rhs || !u || n == 0) {
        return; 
    }

    // Thomas algorithm for symmetric tridiagonal matrix:
    // Diagonal: (1 - 2*w), Off-diagonals: w (both sub and super)
    // This matches the discretization: (1 + 2γΔx⁻²) with off-diagonals -γΔx⁻²
    // where w = -γΔx⁻²
    
    // Forward elimination step
    //DTYPE norm_coeff = 1.0 / (1.0 - 2.0 * w[0]);
    DTYPE norm_coeff;                           
    //tmp[0] = - w[0] * norm_coeff;
    tmp[0] = 0.0;
    //f[0] = f[0] * norm_coeff;
    //rhs[0] = u_boundary - 0.5 * delta_space * (u_BC_derivative_second_direction[0] + u_BC_derivative_third_direction[0]) ;
    rhs[0] = u[0];
    rhs[n-1] = u[n - 1];
    for(int i = 1; i < n - 1; i++){
        norm_coeff = 1.0 / ((1.0 - 2.0 * w[i]) - w[i] * tmp[i - 1]); 
        tmp[i] = w[i] * norm_coeff;
        rhs[i] = (rhs[i] - w[i]*rhs[i - 1]) * norm_coeff;
    }
    //rhs[n-1] = u_BC_current_direction[n-1] - 0.5 * delta_space * (u_BC_derivative_second_direction[n-1] + u_BC_derivative_third_direction[n-1]);
    // Backward substitution
    u[n - 1] = rhs[n - 1];
    for(int i = 1; i < n; i++){
        u[n - 1 - i] = rhs[n - 1 - i] - tmp[n - 1 - i] * u[n - i];
    }
}



void Thomas_Different_Direction(const DTYPE *__restrict__ w, 
                               unsigned int n,
                               DTYPE *__restrict__ tmp,
                               DTYPE *__restrict__ rhs,
                               DTYPE *__restrict__ u,
                               VelocityField u_BC,
                               DTYPE delta_space 

                            ) 
{
    // Check input 
    if (!w || !tmp || !rhs || !u || n == 0) {
        return; 
    }

    // Thomas algorithm for symmetric tridiagonal matrix:
    // Diagonal: (1 - 2*w), Off-diagonals: w (both sub and super)
    // This matches the discretization: (1 + 2γΔx⁻²) with off-diagonals -γΔx⁻²
    // where w = -γΔx⁻²
    
    // Forward elimination step
    //DTYPE norm_coeff = 1.0 / (1.0 - 2.0 * w[0]);
    DTYPE norm_coeff;                           
    //tmp[0] = - w[0] * norm_coeff;
    tmp[0] = 0.0;
    //f[0] = f[0] * norm_coeff;
    //rhs[0] = u_boundary - 0.5 * delta_space * (u_BC_derivative_second_direction[0] + u_BC_derivative_third_direction[0]) ;
    rhs[0] = u[0];
    rhs[n-1] = rhs[n-1] - 2.0 * w[n-1] * u[n-1];
    for(int i = 1; i < n-1; i++){
        norm_coeff = 1.0 / ((1.0 - 2.0 * w[i]) - w[i] * tmp[i - 1]); 
        tmp[i] = w[i] * norm_coeff;
        rhs[i] = (rhs[i] - w[i]*rhs[i - 1]) * norm_coeff;
    }
    norm_coeff = 1.0 / ((1.0 - 3.0 * w[n-1]) - w[n-1] * tmp[n - 2]);
    rhs[n-1] = (rhs[n-1] - w[n-1]*rhs[n-2]) * norm_coeff
    //rhs[n-1] = u_BC_current_direction[n-1] - 0.5 * delta_space * (u_BC_derivative_second_direction[n-1] + u_BC_derivative_third_direction[n-1]);
    // Backward substitution
    u[n - 1] = rhs[n - 1];
    for(int i = 1; i < n; i++){
        u[n - 1 - i] = rhs[n - 1 - i] - tmp[n - 1 - i] * u[n - i];
    }
}



void Thomas_Pressure(const DTYPE *__restrict__ w, 
                                         unsigned int n,
                                         DTYPE *__restrict__ tmp,
                                         DTYPE *__restrict__ f,
                                         DTYPE *__restrict__ u
                                     ) 
{
    // Check input 
    if (!w || !tmp || !f || !u || n == 0) {
        return; 
    }

    // Thomas algorithm for symmetric tridiagonal matrix:
    // Diagonal: (1 - 2*w), Off-diagonals: w (both sub and super)
    // This matches the discretization: (1 + 2γΔx⁻²) with off-diagonals -γΔx⁻²
    // where w = -γΔx⁻²
    
    // Forward elimination step
    DTYPE norm_coeff = 1.0 / (1.0 - 2.0 * w[0]);                           
    tmp[0] = w[0] * norm_coeff;  // Super-diagonal coefficient
    f[0] = f[0] * norm_coeff;
    for(int i = 1; i < n; i++){
        norm_coeff = 1.0 / ((1.0 - 2.0 * w[i]) - w[i] * tmp[i - 1]); 
        tmp[i] = w[i] * norm_coeff;  // Super-diagonal coefficient
        f[i] = (f[i] - w[i]*f[i - 1]) * norm_coeff;  // Sub-diagonal is also w
    }
    // Backward substitution
    u[n - 1] = f[n - 1];
    for(int i = 1; i < n; i++){
        u[n - 1 - i] = f[n - 1 - i] - tmp[n - 1 - i] * u[n - i];
    }
}

void solve_Dxx_tridiag_blocks(DTYPE *Eta_next_component, DTYPE *rhs, DTYPE *Gamma, function v_boundary, bool same_direction){

    // Initialize temporary arrays 
    DTYPE *w = (DTYPE *) malloc(GRID_SIZE * sizeof(DTYPE));
    DTYPE *tmp = (DTYPE *) malloc(GRID_SIZE * sizeof(DTYPE));
    memset(tmp, 0, GRID_SIZE * sizeof(DTYPE));

    // for(int i=0; i< GRID_SIZE; i++){
    //     w[i] = -Gamma[i] * DX_INVERSE_SQUARE;
    // }

    for(int i = 1; i < WIDTH; i++){
        for(int j = 1; j < HEIGHT; j++){
            for(int k = 1; k < DEPTH; k++){
                size_t idx = rowmaj_idx(i,j,k);
                w[idx] = -Gamma[i] * DX_INVERSE_SQUARE;
            }
        }
    }


    if(same_direction){
        /* Solving for each row of the domain, one at a time. */
        int i = 0;
        for (int k = 1; k < DEPTH; k++) {
            for (int j = 1; j < HEIGHT; j++) { //j=1
                /* Here we solve for a single block. */
                size_t off = k * (HEIGHT * WIDTH) + j * WIDTH; //non conta i ghost node
                Thomas_Same_Direction(w + off, WIDTH, tmp + off, rhs + off, Eta_next_component + off,
                                    v_boundary(i, j, k, 0, 0),
                                    DX);
                //Eta_next_component[0]= dirichlet_left(w + off, rhs + off, Eta_next_component + off)
            }
        }
    } else {
        /* Solving for each row of the domain, one at a time. */
        for (int k = 1; k < DEPTH; k++) {
            for (int j = 1; j < HEIGHT; j++) {
                /* Here we solve for a single block. */
                size_t off = k * (HEIGHT * WIDTH) + j * WIDTH; //non conta i ghost node
                Thomas_Different_Direction(w + off, WIDTH, tmp + off, rhs + off, Eta_next_component + off,
                                    v_boundary(i, j, k, 0, 0),
                                    DX);
                //Eta_next_component[0]= dirichlet_left(w + off, f_field_component + off, Eta_next_component + off)
            }
        }
    }




    
    free(w);
    free(tmp);
}

void solve_Dyy_tridiag_blocks(DTYPE *Zeta_next, DTYPE *rhs, DTYPE *Gamma, function v_boundary, bool same_direction){
    // Buffer riutilizzati per ogni colonna (i,k)
    DTYPE *f_block   = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *u_block   = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *w_block   = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *tmp_block = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));

    if (!f_block || !u_block || !w_block || !tmp_block) {
        free(f_block); free(u_block); free(w_block); free(tmp_block);
        return;
    }



    if(same_direction){
        for (int k = 1; k < DEPTH; ++k) {
            for (int i = 1; i < WIDTH; ++i) {
                size_t off = (size_t)k * (HEIGHT * WIDTH) + i; //non conta i ghost node

                // gather lungo y (stride = WIDTH)
                for (int j = 0; j < HEIGHT; ++j){
                    size_t idx = off + (size_t)j * WIDTH;
                    rhs_block[j] = rhs[idx];
                    w_block[j] = - Gamma[idx] * DY_INVERSE_SQUARE;
                }

                // Risolve A_y u = f con algoritmo di Thomas
                Thomas_Same_Direction(w_block, HEIGHT, tmp_block, rhs_block, u_block,
                                    v_boundary(i, HEIGHT, k, 0, 1),
                                    DY);

                // scatter risultato
                for (int j = 0; j < HEIGHT; ++j){
                    size_t idx = off + (size_t)j * WIDTH;
                    Zeta_next[idx] = u_block[j];
                }
            }
        }
    } else {    
        for (int k = 1; k < DEPTH; ++k) {
            for (int i = 1; i < WIDTH; ++i) {
                size_t off = (size_t)k * (HEIGHT * WIDTH) + i; //non conta i ghost node

                // gather lungo y (stride = WIDTH)
                for (int j = 0; j < HEIGHT; ++j){
                    size_t idx = off + (size_t)j * WIDTH;
                    rhs_block[j] = rhs[idx];
                    w_block[j] = - Gamma[idx] * DY_INVERSE_SQUARE;
                }

                // Risolve A_y u = f con algoritmo di Thomas
                Thomas_Different_Direction(w_block, HEIGHT, tmp_block, rhs_block, u_block,
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
}

void solve_Dzz_tridiag_blocks(DTYPE *U_next, DTYPE *rhs, DTYPE *Gamma, function v_boundary, bool same_direction){
    // Buffer riutilizzati per ogni colonna (i,k)
    DTYPE *f_block   = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *u_block   = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *w_block   = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *tmp_block = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));

    if (!f_block || !u_block || !w_block || !tmp_block) {
        free(f_block); free(u_block); free(w_block); free(tmp_block);
        return;
    }

    if(same_direction){
        for (int j = 1; j < HEIGHT; ++j) {
            for (int i = 1; i < WIDTH; ++i) {
                size_t off = (size_t)j * WIDTH + i;

                // gather lungo z (stride = HEIGHT * WIDTH)
                for (int k = 0; k < DEPTH; ++k){
                    size_t idx = off + (size_t)k * (HEIGHT * WIDTH);
                    rhs_block[k] = rhs[idx];
                    w_block[k] = - Gamma[idx] * DZ_INVERSE_SQUARE;
                }

                // Risolve A_z u = f con algoritmo di Thomas
                Thomas_Same_Direction(w_block, DEPTH, tmp_block, rhs_block, u_block,
                                    u_BC, DZ);

                // scatter risultato
                for (int k = 0; k < DEPTH; ++k){
                    size_t idx = off + (size_t)k * (HEIGHT * WIDTH);
                    U_next[idx] = u_block[k];
                }
            }
        }
    } else {
        for (int j = 1; j < HEIGHT; ++j) {
            for (int i = 1; i < WIDTH; ++i) {
                size_t off = (size_t)j * WIDTH + i;

                // gather lungo z (stride = HEIGHT * WIDTH)
                for (int k = 0; k < DEPTH; ++k){
                    size_t idx = off + (size_t)k * (HEIGHT * WIDTH);
                    rhs_block[k] = rhs[idx];
                    w_block[k] = - Gamma[idx] * DZ_INVERSE_SQUARE;
                }

                // Risolve A_z u = f con algoritmo di Thomas
                Thomas_Different_Direction(w_block, DEPTH, tmp_block, rhs_block, u_block,
                                    u_BC, DZ);

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
}