#include "tridiagonal_blocks.h"

static void Thomas(const DTYPE *__restrict__ w, 
                               unsigned int n,
                               DTYPE *__restrict__ tmp,
                               DTYPE *__restrict__ f,
                               DTYPE *__restrict__ u) 
{
    // Check input 
    if (!w || !tmp || !f || !u || n == 0) {
        return; 
    }

    // Forward elimination step
    DTYPE norm_coeff = 1 / (1 - 2 * w[0]);                           
    tmp[0] = - w[0] * norm_coeff;
    f[0] = f[0] * norm_coeff;
    for(int i = 1; i < n; i++){
        norm_coeff = 1 / ((1 - 2 * w[i]) - w[i] * tmp[i - 1]); 
        tmp[i] = -w[i] * norm_coeff;
        f[i] = (f[i] + w[i]*f[i - 1]) * norm_coeff;
    }
    
    // Backward substitution
    u[n - 1] = f[n - 1];
    for(int i = 1; i < n; i++){
        u[n - 1 - i] = -u[n - i]*tmp[n - 1 - i] + f[n - 1 - i];
    }
}

void solve_Dxx_tridiag_blocks(DTYPE *Eta_next_component, DTYPE *f_field_component, DTYPE *Gamma){
    // Initialize temporary arrays 
    DTYPE *w = (DTYPE *) malloc(GRID_SIZE * sizeof(DTYPE));
    DTYPE *tmp = (DTYPE *) malloc(GRID_SIZE * sizeof(DTYPE));
    memset(tmp, 0, GRID_SIZE * sizeof(DTYPE));

    for(int i=0; i< GRID_SIZE; i++){
        w[i] = -Gamma[i] * DX_INVERSE_SQUARE;
    }

    /* Solving for each row of the domain, one at a time. */
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            /* Here we solve for a single block. */
            size_t off = k * (HEIGHT * WIDTH) + j * WIDTH;
            Thomas(w + off, WIDTH, tmp, f_field_component + off, Eta_next_component + off);
        }
    }
    
    free(w);
    free(tmp);
}

void solve_Dyy_tridiag_blocks(DTYPE *Zeta_next, DTYPE *f_field, DTYPE *Gamma){
    // Buffer riutilizzati per ogni colonna (i,k)
    DTYPE *f_block   = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *u_block   = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *w_block   = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *tmp_block = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));

    if (!f_block || !u_block || !w_block || !tmp_block) {
        free(f_block); free(u_block); free(w_block); free(tmp_block);
        return;
    }

    for (int k = 0; k < DEPTH; ++k) {
        for (int i = 0; i < WIDTH; ++i) {
            size_t off = (size_t)k * (HEIGHT * WIDTH) + i;

            // gather lungo y (stride = WIDTH)
            for (int j = 0; j < HEIGHT; ++j){
                size_t idx = off + (size_t)j * WIDTH;
                f_block[j] = f_field[idx];
                w_block[j] = - Gamma[idx] * DY_INVERSE_SQUARE;
            }

            // Risolve A_y u = f con algoritmo di Thomas
            Thomas(w_block, HEIGHT, tmp_block, f_block, u_block);

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
}

void solve_Dzz_tridiag_blocks(DTYPE *U_next, DTYPE *f_field, DTYPE *Gamma){
    // Buffer riutilizzati per ogni colonna (i,k)
    DTYPE *f_block   = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *u_block   = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *w_block   = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
    DTYPE *tmp_block = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));

    if (!f_block || !u_block || !w_block || !tmp_block) {
        free(f_block); free(u_block); free(w_block); free(tmp_block);
        return;
    }

    for (int j = 0; j < HEIGHT; ++j) {
        for (int i = 0; i < WIDTH; ++i) {
            size_t off = (size_t)j * WIDTH + i;

            // gather lungo z (stride = HEIGHT * WIDTH)
            for (int k = 0; k < DEPTH; ++k){
                size_t idx = off + (size_t)k * (HEIGHT * WIDTH);
                f_block[k] = f_field[idx];
                w_block[k] = - Gamma[idx] * DZ_INVERSE_SQUARE;
            }

            // Risolve A_z u = f con algoritmo di Thomas
            Thomas(w_block, DEPTH, tmp_block, f_block, u_block);

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
}