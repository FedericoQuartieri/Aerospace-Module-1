#include "momentum.h"


/* eta: rhs = u + (DT/beta)*g - eta 
 * v_comp = [x:0, y:1, z:2]
 * */
void update_eta(SolverMemState *solver_mem_state,
                Real *restrict rhs,
                Real *restrict tmp,
                Data *data, int t_step, int v_comp) {
        
    const Real *restrict k_porosity;
    const Real *restrict u;
    Real *eta;

    switch(v_comp) {
        case 0:
            k_porosity = solver_mem_state->k.v_x;
            u = solver_mem_state->u.v_x;
            eta = solver_mem_state->eta.v_x;
            break;
        case 1:
            k_porosity = solver_mem_state->k.v_y;
            u = solver_mem_state->u.v_y;
            eta = solver_mem_state->eta.v_y;
            break;
        case 2:
            k_porosity = solver_mem_state->k.v_z;
            u = solver_mem_state->u.v_z;
            eta = solver_mem_state->eta.v_z;
            break;
        default: 
            fprintf(stderr, "Value of v_comp doesn't exist");
            exit(1);    
    }

    const bool same_direction = (v_comp == 0);

    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            // Row offset index
            const size_t off = (size_t)k * (WIDTH * HEIGHT)
                             + (size_t)j * WIDTH;
            
            // Thomas algorithm Dxx
            tmp[0] = 0.0;
            rhs[0] = bc_left(data->bc_velocity, 0, j, k, t_step, v_comp);
            
            // Forwards step
            for (int i = 1; i < WIDTH-1; i++) {
                const Real k_i = k_porosity[off+i];
                const Real w_i = -gamma_from_k(k_i) * DX_INVERSE_SQUARE;
                
                const Real norm_coeff =
                    1.0 / ((1.0 - 2.0 * w_i) - w_i * tmp[i - 1]);
                tmp[i] = w_i * norm_coeff;
               
                const Real beta_i = beta_from_k(k_i);
                const Real xi_i =
                    u[off+i] + (DT/beta_i)*g_value(i, j, k, t_step, k_i,
                                                 solver_mem_state, data,
                                                 v_comp);
                rhs[i] = xi_i - eta[off + i];
                rhs[i] = (rhs[i] - w_i * rhs[i - 1]) * norm_coeff;
            }

            //Compute last rhs value
            const Real k_i = k_porosity[off+WIDTH-1];
            const Real w_i = -gamma_from_k(k_i) * DX_INVERSE_SQUARE;
            const Real beta_i = beta_from_k(k_i);
            const Real xi_i =
                u[off+WIDTH-1]
                + (DT/beta_i)*g_value(WIDTH-1, j, k, t_step, k_i,
                                      solver_mem_state, data, v_comp);
            rhs[WIDTH-1] = xi_i - eta[off+WIDTH-1];
            const Real right_value =
                bc_right(data->bc_velocity, WIDTH-1, j, k, t_step, v_comp);
            rhs[WIDTH-1] = rhs[WIDTH-1] - 2.0 * w_i * right_value;
            
            const Real norm_coeff =
                1.0 / ((1.0 - 3.0 * w_i) - w_i * tmp[WIDTH - 2]);
            rhs[WIDTH-1] = (rhs[WIDTH-1] - w_i*rhs[WIDTH-2]) * norm_coeff;

            // Last value of rhs depends on same_direction
            Real up1 = same_direction ? right_value : rhs[WIDTH - 1];
            
            // Backwards step
            eta[off+WIDTH-1] += up1;
            for (int i = WIDTH-2; i >= 0; i--) {
                const Real up2 = rhs[i] - tmp[i]*up1;
                eta[off + i] += up2;
                up1 = up2;
            }
        }
    }
}

/* zeta:
 * rhs = eta - zeta
 */
void update_zeta(SolverMemState *solver_mem_state,
                 Real *restrict rhs,
                 Real *restrict tmp,
                 Data *data, int t_step, int v_comp) {
    const Real *restrict k_porosity;
    const Real *restrict eta;
    Real *restrict zeta;

    switch(v_comp) {
        case 0:
            k_porosity = solver_mem_state->k.v_x;
            zeta = solver_mem_state->zeta.v_x;
            eta = solver_mem_state->eta.v_x;
            break;
        case 1:
            k_porosity = solver_mem_state->k.v_y;
            zeta = solver_mem_state->zeta.v_y;
            eta = solver_mem_state->eta.v_y;
            break;
        case 2:
            k_porosity = solver_mem_state->k.v_z;
            zeta = solver_mem_state->zeta.v_z;
            eta = solver_mem_state->eta.v_z;
            break;
        default: 
            fprintf(stderr, "Value of v_comp doesn't exist");
            exit(1);    
    }

   const bool same_direction = (v_comp == 1);

   for (int k = 0; k < DEPTH; k++) {
       for (int i = 0; i < WIDTH; i++) {
            const size_t off = k * (WIDTH*HEIGHT) + i;
            
            tmp[0] = 0.0;
            rhs[0] = bc_left(data->bc_velocity, i, 0, k, t_step, v_comp);
            
            for (int j = 1; j < HEIGHT-1; j++) {
                const size_t j_arr = off + j*WIDTH;
                
                const Real k_i = k_porosity[j_arr];
                const Real w_i = -gamma_from_k(k_i) * DY_INVERSE_SQUARE;
                
                const Real norm_coeff =
                    1.0 / ((1.0 - 2.0 * w_i) - w_i * tmp[j - 1]);
                tmp[j] = w_i * norm_coeff;
                
                rhs[j] = eta[j_arr] - zeta[j_arr];
                rhs[j] = (rhs[j] - w_i * rhs[j - 1]) * norm_coeff;
            }

             //Compute last rhs value
            size_t j_arr = off + (HEIGHT-1)*WIDTH;
            const Real k_i = k_porosity[j_arr];
            const Real w_i = -gamma_from_k(k_i) * DY_INVERSE_SQUARE;
            
            rhs[HEIGHT-1] = eta[j_arr] - zeta[j_arr];
            const Real right_value =
                bc_right(data->bc_velocity, i, HEIGHT-1, k, t_step, v_comp);
            rhs[HEIGHT-1] = rhs[HEIGHT-1] - 2.0 * w_i * right_value;
            
            const Real norm_coeff =
                1.0 / ((1.0 - 3.0 * w_i) - w_i * tmp[HEIGHT - 2]);
            rhs[HEIGHT-1] = (rhs[HEIGHT-1] - w_i*rhs[HEIGHT-2]) * norm_coeff;

            // Last value of rhs depends on same_direction
            Real up1 = same_direction ? right_value : rhs[HEIGHT - 1];
            
            // Backwards step
            zeta[j_arr] += up1;
            for (int j = HEIGHT-2; j >= 0; j--) {
                j_arr = off + j*WIDTH;
                const Real up2 = rhs[j] - tmp[j]*up1;
                zeta[j_arr] += up2;
                up1 = up2;
            }
       }
   }
}

/* u:
 * rhs = zeta - u
 */
void update_u(SolverMemState *solver_mem_state,
              Real *restrict rhs,
              Real *restrict tmp,
              Data *data, int t_step, int v_comp) {
    const Real *restrict k_porosity;
    const Real *restrict zeta;
    Real *restrict u;

    switch(v_comp) {
        case 0:
            k_porosity = solver_mem_state->k.v_x;
            zeta = solver_mem_state->zeta.v_x;
            u = solver_mem_state->u.v_x;
            break;
        case 1:
            k_porosity = solver_mem_state->k.v_y;
            zeta = solver_mem_state->zeta.v_y;
            u = solver_mem_state->u.v_y;
            break;
        case 2:
            k_porosity = solver_mem_state->k.v_z;
            zeta = solver_mem_state->zeta.v_z;
            u = solver_mem_state->u.v_z;
            break;
        default:
            fprintf(stderr, "Value of v_comp doesn't exist");
            exit(1);
    }

    const bool same_direction = (v_comp == 2);
    const size_t plane_size = (size_t)WIDTH * HEIGHT;

    for (int j = 0; j < HEIGHT; j++) {
        for (int i = 0; i < WIDTH; i++) {
            const size_t off = (size_t)j * WIDTH + i;

            tmp[0] = 0.0;
            rhs[0] =
                bc_left(data->bc_velocity, i, j, 0, t_step, v_comp);

            for (int k = 1; k < DEPTH-1; k++) {
                const size_t k_arr = off + (size_t)k * plane_size;

                const Real k_i = k_porosity[k_arr];
                const Real w_i =
                    -gamma_from_k(k_i) * DZ_INVERSE_SQUARE;

                const Real norm_coeff =
                    1.0 / ((1.0 - 2.0 * w_i) - w_i * tmp[k - 1]);
                tmp[k] = w_i * norm_coeff;

                rhs[k] = zeta[k_arr] - u[k_arr];
                rhs[k] =
                    (rhs[k] - w_i * rhs[k - 1]) * norm_coeff;
            }

            // Compute last rhs value
            size_t k_arr = off + (size_t)(DEPTH-1) * plane_size;
            const Real k_i = k_porosity[k_arr];
            const Real w_i =
                -gamma_from_k(k_i) * DZ_INVERSE_SQUARE;

            rhs[DEPTH-1] = zeta[k_arr] - u[k_arr];
            const Real right_value =
                bc_right(data->bc_velocity, i, j, DEPTH-1,
                         t_step, v_comp);
            rhs[DEPTH-1] =
                rhs[DEPTH-1] - 2.0 * w_i * right_value;

            const Real norm_coeff =
                1.0 / ((1.0 - 3.0 * w_i)
                       - w_i * tmp[DEPTH - 2]);
            rhs[DEPTH-1] =
                (rhs[DEPTH-1] - w_i * rhs[DEPTH-2]) * norm_coeff;

            // Last value of rhs depends on same_direction
            Real up1 =
                same_direction ? right_value : rhs[DEPTH - 1];

            // Backwards step
            u[k_arr] += up1;
            for (int k = DEPTH-2; k >= 0; k--) {
                k_arr = off + (size_t)k * plane_size;
                const Real up2 = rhs[k] - tmp[k]*up1;
                u[k_arr] += up2;
                up1 = up2;
            }
        }
    }
}

void momentum_step(SolverMemState *solver_mem_state,
                   Real *restrict rhs,
                   Real *restrict tmp,
                   Data *data, int t_step, SolverStats *solver_stats) {
    
    // eta: compute next update for the three component
    uint64_t start_ns = time_ns();
    update_eta(solver_mem_state, rhs, tmp, data, t_step, 0);
    update_eta(solver_mem_state, rhs, tmp, data, t_step, 1);
    update_eta(solver_mem_state, rhs, tmp, data, t_step, 2);
    solver_stats->eta_sys += time_ns() - start_ns;

    // zeta: compute next update for the three component
    start_ns = time_ns();
    update_zeta(solver_mem_state, rhs, tmp, data, t_step, 0);
    update_zeta(solver_mem_state, rhs, tmp, data, t_step, 1);
    update_zeta(solver_mem_state, rhs, tmp, data, t_step, 2);
    solver_stats->zeta_sys += time_ns() - start_ns;

    // u: compute next update for the three component
    start_ns = time_ns();
    update_u(solver_mem_state, rhs, tmp, data, t_step, 0);
    update_u(solver_mem_state, rhs, tmp, data, t_step, 1);
    update_u(solver_mem_state, rhs, tmp, data, t_step, 2);
    solver_stats->u_sys += time_ns() - start_ns;
}
