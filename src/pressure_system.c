#include "pressure_system.h"
#include "velocity_field.h"

void solve_pressure_system(VelocityField U_next, 
                           Pressure *psi, 
                           Pressure *phi_lower, 
                           Pressure *phi_higher,
                           Pressure *pressure
                        )
{
                            
    compute_Psi(U_next, psi);
    compute_Phi_lower(psi,  phi_lower);
    compute_Phi_higher(phi_lower, phi_higher);
    compute_pressure(phi_higher, pressure);
}

static void compute_Psi(VelocityField U_next, Pressure *psi){
    // Initialize temporary arrays 
    DTYPE *w = (DTYPE *) malloc(GRID_SIZE * sizeof(DTYPE));
    DTYPE *tmp = (DTYPE *) malloc(GRID_SIZE * sizeof(DTYPE));
    memset(tmp, 0, GRID_SIZE * sizeof(DTYPE));

    Pressure rhs;
    initialize_pressure(&rhs);
     for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                rhs.p[idx] = (compute_velocity_x_grad(U_next.v_x, i, j, k) +
                              compute_velocity_y_grad(U_next.v_y, i, j, k) +
                              compute_velocity_z_grad(U_next.v_z, i, j, k)) *  (-1.0 /DT);

                w[idx] = - DX_INVERSE_SQUARE;
            }
        }
    }

    /* Solving for each row of the domain, one at a time. */
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            /* Here we solve for a single block. */
            size_t off = k * (HEIGHT * WIDTH) + j * WIDTH;

            Thomas_Pressure(w + off, WIDTH, tmp, rhs.p + off, psi->p + off);
        }
    }

    free(tmp);
    free(w);
    free_pressure(&rhs);
}

static void compute_Phi_lower(Pressure *psi, Pressure *phi_lower){
    // Initialize temporary arrays 
    DTYPE *w = (DTYPE *) malloc(GRID_SIZE * sizeof(DTYPE));
    DTYPE *w_block = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *rhs_block = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *u_block = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    DTYPE *tmp_thomas = (DTYPE *) malloc(HEIGHT * sizeof(DTYPE));
    
    Pressure rhs;
        initialize_pressure(&rhs);
        for(int k = 0; k < DEPTH; k++){
            for(int j = 0; j < HEIGHT; j++){
                for(int i = 0; i < WIDTH; i++){
                    size_t idx = rowmaj_idx(i,j,k);

                    rhs.p[idx] = psi->p[idx];
                    w[idx] = - DY_INVERSE_SQUARE;
                }
            }
        }

    // Loop sui sistemi 1D lungo Y (su ogni colonna i,k)
    for (int k = 0; k < DEPTH; ++k) {
        for (int i = 0; i < WIDTH; ++i) {
            
            size_t off = (size_t)k * (HEIGHT * WIDTH) + i; // Offset per la colonna (i,k)

            // 1. GATHER (Raccogli i dati lungo Y)
            for (int j = 0; j < HEIGHT; ++j){
                size_t idx = off + (size_t)j * WIDTH; // Indice 3D
                
                // Assumendo il passo di PRESSIONE (no Gamma)
                rhs_block[j] = rhs.p[idx]; 
                w_block[j] = w[idx]; // w[idx] è -DY_INVERSE_SQUARE
            }

            Thomas_Pressure(w_block, HEIGHT, tmp_thomas, rhs_block, u_block);

            // 3. SCATTER (Spargi il risultato in phi_lower)
            for (int j = 0; j < HEIGHT; ++j){
                size_t idx = off + (size_t)j * WIDTH; // Indice 3D
                
                phi_lower->p[idx] = u_block[j]; // Scrivi nell'array 3D di output
            }
        }
    }

    free(tmp_thomas);
    free(rhs_block);
    free(u_block);
    free(w);
    free(w_block);
    free_pressure(&rhs);
};
                        
static void compute_Phi_higher(Pressure *phi_lower, Pressure *phi_higher){
    DTYPE *w = (DTYPE *) malloc(GRID_SIZE * sizeof(DTYPE));
    DTYPE *w_block = (DTYPE *) malloc(DEPTH * sizeof(DTYPE));
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
                    w[idx] = - DZ_INVERSE_SQUARE;
                }
            }
        }

    // Loop sui sistemi 1D lungo Y (su ogni colonna i,k)
    for (int j = 0; j < HEIGHT; ++j) {
        for (int i = 0; i < WIDTH; ++i) {
            
            size_t off = (size_t)j * WIDTH + i;

            // 1. GATHER (Raccogli i dati lungo Y)
            for (int k = 0; k < DEPTH; ++k){
                size_t idx = off + (size_t)k * (HEIGHT * WIDTH); // Indice 3D
                
                // Assumendo il passo di PRESSIONE (no Gamma)
                rhs_block[k] = rhs.p[idx]; 
                w_block[k] = w[idx];
            }

            Thomas_Pressure(w_block, DEPTH, tmp_thomas, rhs_block, u_block);

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
    free(w);
    free(w_block);
    free_pressure(&rhs);
};

static void compute_pressure(Pressure *phi_higher, Pressure *pressure){
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                pressure->p[idx] += phi_higher->p[idx];
            }
        }
    }
};