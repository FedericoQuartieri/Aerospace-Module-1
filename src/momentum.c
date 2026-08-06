#include "momentum.h"


/* eta: rhs = u + (DT/beta)*g - eta
 * v_comp = [x:0, y:1, z:2]
 * */
void update_eta(const Decomp *d,
                SolverMemState *solver_mem_state,
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

    bool same_direction = (v_comp == 0);
    /* X is the contiguous direction, so a line advances one element at a time. */
    const int nx = d->n[0];

    for (int k = 0; k < d->n[2]; k++) {
        int gk = decomp_global(d, k, 2);

        for (int j = 0; j < d->n[1]; j++) {
            int gj = decomp_global(d, j, 1);
            // Row offset index
            size_t off = decomp_index(d, 0, j, k);

            // Thomas algorithm Dxx
            tmp[0] = 0.0;
            rhs[0] = bc_left(data->bc_velocity,
                             decomp_global(d, 0, 0), gj, gk,
                             t_step, v_comp);

            // Forwards step
            for (int i = 1; i < nx-1; i++) {
                Real k_i = k_porosity[off+i];
                Real w_i = -gamma_from_k(k_i) * DX_INVERSE_SQUARE;

                Real norm_coeff =
                    1.0 / ((1.0 - 2.0 * w_i) - w_i * tmp[i - 1]);
                tmp[i] = w_i * norm_coeff;

                Real beta_i = beta_from_k(k_i);
                Real xi_i =
                    u[off+i] + (DT/beta_i)*g_value(d, i, j, k, t_step, k_i,
                                                 solver_mem_state, data,
                                                 v_comp);
                rhs[i] = xi_i - eta[off + i];
                rhs[i] = (rhs[i] - w_i * rhs[i - 1]) * norm_coeff;
            }

            //Compute last rhs value
            Real k_i = k_porosity[off+nx-1];
            Real w_i = -gamma_from_k(k_i) * DX_INVERSE_SQUARE;
            Real beta_i = beta_from_k(k_i);
            Real xi_i =
                u[off+nx-1]
                + (DT/beta_i)*g_value(d, nx-1, j, k, t_step, k_i,
                                      solver_mem_state, data, v_comp);
            rhs[nx-1] = xi_i - eta[off+nx-1];
            Real right_value =
                bc_right(data->bc_velocity,
                         decomp_global(d, nx-1, 0), gj, gk,
                         t_step, v_comp);
            rhs[nx-1] = rhs[nx-1] - 2.0 * w_i * right_value;

            Real norm_coeff =
                1.0 / ((1.0 - 3.0 * w_i) - w_i * tmp[nx - 2]);
            rhs[nx-1] = (rhs[nx-1] - w_i*rhs[nx-2]) * norm_coeff;

            // Last value of rhs depends on same_direction
            Real up1 = same_direction ? right_value : rhs[nx - 1];

            // Backwards step
            eta[off+nx-1] += up1;
            for (int i = nx-2; i >= 0; i--) {
                Real up2 = rhs[i] - tmp[i]*up1;
                eta[off + i] += up2;
                up1 = up2;
            }
        }
    }
}

/* zeta:
 * rhs = eta - zeta
 */
void update_zeta(const Decomp *d,
                 SolverMemState *solver_mem_state,
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

   bool same_direction = (v_comp == 1);
   const int ny = d->n[1];
   const size_t stride_y = d->stride[1];

   for (int k = 0; k < d->n[2]; k++) {
       int gk = decomp_global(d, k, 2);

       for (int i = 0; i < d->n[0]; i++) {
            int gi = decomp_global(d, i, 0);
            size_t off = decomp_index(d, i, 0, k);

            tmp[0] = 0.0;
            rhs[0] = bc_left(data->bc_velocity,
                             gi, decomp_global(d, 0, 1), gk,
                             t_step, v_comp);

            for (int j = 1; j < ny-1; j++) {
                size_t j_arr = off + (size_t)j * stride_y;

                Real k_i = k_porosity[j_arr];
                Real w_i = -gamma_from_k(k_i) * DY_INVERSE_SQUARE;

                Real norm_coeff =
                    1.0 / ((1.0 - 2.0 * w_i) - w_i * tmp[j - 1]);
                tmp[j] = w_i * norm_coeff;

                rhs[j] = eta[j_arr] - zeta[j_arr];
                rhs[j] = (rhs[j] - w_i * rhs[j - 1]) * norm_coeff;
            }

             //Compute last rhs value
            size_t j_arr = off + (size_t)(ny-1) * stride_y;
            Real k_i = k_porosity[j_arr];
            Real w_i = -gamma_from_k(k_i) * DY_INVERSE_SQUARE;

            rhs[ny-1] = eta[j_arr] - zeta[j_arr];
            Real right_value =
                bc_right(data->bc_velocity,
                         gi, decomp_global(d, ny-1, 1), gk,
                         t_step, v_comp);
            rhs[ny-1] = rhs[ny-1] - 2.0 * w_i * right_value;

            Real norm_coeff =
                1.0 / ((1.0 - 3.0 * w_i) - w_i * tmp[ny - 2]);
            rhs[ny-1] = (rhs[ny-1] - w_i*rhs[ny-2]) * norm_coeff;

            // Last value of rhs depends on same_direction
            Real up1 = same_direction ? right_value : rhs[ny - 1];

            // Backwards step
            zeta[j_arr] += up1;
            for (int j = ny-2; j >= 0; j--) {
                j_arr = off + (size_t)j * stride_y;
                Real up2 = rhs[j] - tmp[j]*up1;
                zeta[j_arr] += up2;
                up1 = up2;
            }
       }
   }
}

/* u:
 * rhs = zeta - u
 */
void update_u(const Decomp *d,
              SolverMemState *solver_mem_state,
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

    bool same_direction = (v_comp == 2);
    const int nz = d->n[2];
    const size_t stride_z = d->stride[2];

    for (int j = 0; j < d->n[1]; j++) {
        int gj = decomp_global(d, j, 1);

        for (int i = 0; i < d->n[0]; i++) {
            int gi = decomp_global(d, i, 0);
            size_t off = decomp_index(d, i, j, 0);

            tmp[0] = 0.0;
            rhs[0] =
                bc_left(data->bc_velocity,
                        gi, gj, decomp_global(d, 0, 2),
                        t_step, v_comp);

            for (int k = 1; k < nz-1; k++) {
                size_t k_arr = off + (size_t)k * stride_z;

                Real k_i = k_porosity[k_arr];
                Real w_i =
                    -gamma_from_k(k_i) * DZ_INVERSE_SQUARE;

                Real norm_coeff =
                    1.0 / ((1.0 - 2.0 * w_i) - w_i * tmp[k - 1]);
                tmp[k] = w_i * norm_coeff;

                rhs[k] = zeta[k_arr] - u[k_arr];
                rhs[k] =
                    (rhs[k] - w_i * rhs[k - 1]) * norm_coeff;
            }

            // Compute last rhs value
            size_t k_arr = off + (size_t)(nz-1) * stride_z;
            Real k_i = k_porosity[k_arr];
            Real w_i =
                -gamma_from_k(k_i) * DZ_INVERSE_SQUARE;

            rhs[nz-1] = zeta[k_arr] - u[k_arr];
            Real right_value =
                bc_right(data->bc_velocity,
                         gi, gj, decomp_global(d, nz-1, 2),
                         t_step, v_comp);
            rhs[nz-1] =
                rhs[nz-1] - 2.0 * w_i * right_value;

            Real norm_coeff =
                1.0 / ((1.0 - 3.0 * w_i)
                       - w_i * tmp[nz - 2]);
            rhs[nz-1] =
                (rhs[nz-1] - w_i * rhs[nz-2]) * norm_coeff;

            // Last value of rhs depends on same_direction
            Real up1 =
                same_direction ? right_value : rhs[nz - 1];

            // Backwards step
            u[k_arr] += up1;
            for (int k = nz-2; k >= 0; k--) {
                k_arr = off + (size_t)k * stride_z;
                Real up2 = rhs[k] - tmp[k]*up1;
                u[k_arr] += up2;
                up1 = up2;
            }
        }
    }
}

void momentum_step(const Decomp *decomp,
                   SolverMemState *solver_mem_state,
                   Real *restrict rhs,
                   Real *restrict tmp,
                   Data *data, int t_step, SolverStats *solver_stats) {

    // eta: compute next update for the three component
    uint64_t start_ns = time_ns();
    update_eta(decomp, solver_mem_state, rhs, tmp, data, t_step, 0);
    update_eta(decomp, solver_mem_state, rhs, tmp, data, t_step, 1);
    update_eta(decomp, solver_mem_state, rhs, tmp, data, t_step, 2);
    solver_stats->eta_sys += time_ns() - start_ns;

    // zeta: compute next update for the three component
    start_ns = time_ns();
#if defined(USE_SIMD) && SIMD_AVAILABLE
    update_zeta_simd(decomp, solver_mem_state, rhs, tmp, data, t_step, 0,
                     ZETA_SIMD_LINES);
    update_zeta_simd(decomp, solver_mem_state, rhs, tmp, data, t_step, 1,
                     ZETA_SIMD_LINES);
    update_zeta_simd(decomp, solver_mem_state, rhs, tmp, data, t_step, 2,
                     ZETA_SIMD_LINES);
#else
    update_zeta(decomp, solver_mem_state, rhs, tmp, data, t_step, 0);
    update_zeta(decomp, solver_mem_state, rhs, tmp, data, t_step, 1);
    update_zeta(decomp, solver_mem_state, rhs, tmp, data, t_step, 2);
#endif
    solver_stats->zeta_sys += time_ns() - start_ns;

    // u: compute next update for the three component
    start_ns = time_ns();
#if defined(USE_SIMD) && SIMD_AVAILABLE
    update_u_simd(decomp, solver_mem_state, rhs, tmp, data, t_step, 0,
                  U_SIMD_LINES);
    update_u_simd(decomp, solver_mem_state, rhs, tmp, data, t_step, 1,
                  U_SIMD_LINES);
    update_u_simd(decomp, solver_mem_state, rhs, tmp, data, t_step, 2,
                  U_SIMD_LINES);
#else
    update_u(decomp, solver_mem_state, rhs, tmp, data, t_step, 0);
    update_u(decomp, solver_mem_state, rhs, tmp, data, t_step, 1);
    update_u(decomp, solver_mem_state, rhs, tmp, data, t_step, 2);
#endif
    solver_stats->u_sys += time_ns() - start_ns;
}
