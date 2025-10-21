#include "tridiagonal_blocks.h"

static void solve_Dxx_tridiag(const DTYPE *__restrict__ w, 
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

void solve_Dxx_tridiag_blocks(DTYPE *Eta_next_component, DTYPE *f_field_component){
    // Initialize temporary arrays 
    DTYPE *w = (DTYPE *) malloc(GRID_SIZE);
    DTYPE *tmp = (DTYPE *) malloc(GRID_SIZE);
    memset(w,0,GRID_SIZE);
    memset(tmp,0,GRID_SIZE);

    /* Solving for each row of the domain, one at a time. */
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            /* Here we solve for a single block. */
            size_t off = k * (HEIGHT * WIDTH) + j * WIDTH;
            solve_Dxx_tridiag(w + off, WIDTH, tmp, f_field_component + off, Eta_next_component + off);
        }
    }
    
    free(w);
    free(tmp);
}


