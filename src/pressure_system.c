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
}

static void compute_Phi_lower(Pressure *psi, Pressure *phi_lower){};
                        
static void compute_Phi_higher(Pressure *phi_lower, Pressure *phi_higher){};

static void compute_pressure(Pressure *phi_higher, Pressure *pressure){};