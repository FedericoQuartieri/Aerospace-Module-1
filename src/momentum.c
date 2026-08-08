#include "momentum.h"
#include "schur.h"


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

/*
 * u: rhs = zeta - u, solved along Z.
 *
 * Unlike eta and zeta, this direction is the one the grid is split along, so
 * the line a process owns is only a piece of the real one.  Instead of running
 * Thomas on it, the three diagonals and the right-hand side are written out
 * explicitly and handed to schur_solve_mpi, which stitches the pieces back
 * together.  With a single process it falls back to plain Thomas, so nothing
 * changes there.
 *
 * The systems are built one XY row at a time: all the lines of a row are
 * solved together, so the whole row costs one exchange instead of one per
 * line, and the buffers stay small.
 */
void update_u(const Decomp *d,
              SolverMemState *solver_mem_state,
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
    const int nx = d->n[0];
    const int nz = d->n[2];
    const int last_global = d->n_global[2] - 1;
    const size_t stride_z = d->stride[2];

    size_t room = (size_t)nx * (size_t)nz * sizeof(Real);
    Real *lower = xmalloc(room);
    Real *diagonal = xmalloc(room);
    Real *upper = xmalloc(room);
    Real *known = xmalloc(room);
    Real *increment = xmalloc(room);

    for (int j = 0; j < d->n[1]; j++) {
        int gj = decomp_global(d, j, 1);

        for (int i = 0; i < nx; i++) {
            int gi = decomp_global(d, i, 0);
            size_t column = decomp_index(d, i, j, 0);
            size_t line = (size_t)i * (size_t)nz;

            for (int k = 0; k < nz; k++) {
                int gk = decomp_global(d, k, 2);
                size_t cell = column + (size_t)k * stride_z;
                size_t at = line + (size_t)k;
                Real w_i = -gamma_from_k(k_porosity[cell]) * DZ_INVERSE_SQUARE;

                if (gk == 0) {
                    /* Parete inferiore: il valore e' imposto. */
                    lower[at] = 0.0;
                    diagonal[at] = 1.0;
                    upper[at] = 0.0;
                    known[at] = bc_left(data->bc_velocity, gi, gj, gk,
                                        t_step, v_comp);
                } else if (gk == last_global) {
                    Real right_value =
                        bc_right(data->bc_velocity, gi, gj, gk,
                                 t_step, v_comp);

                    if (same_direction) {
                        /* Componente normale alla parete: imposta anch'essa. */
                        lower[at] = 0.0;
                        diagonal[at] = 1.0;
                        upper[at] = 0.0;
                        known[at] = right_value;
                    } else {
                        /* Componente tangente: nodo fantasma eliminato. */
                        lower[at] = w_i;
                        diagonal[at] = 1.0 - 3.0 * w_i;
                        upper[at] = 0.0;
                        known[at] = zeta[cell] - u[cell]
                                    - 2.0 * w_i * right_value;
                    }
                } else {
                    lower[at] = w_i;
                    diagonal[at] = 1.0 - 2.0 * w_i;
                    upper[at] = w_i;
                    known[at] = zeta[cell] - u[cell];
                }
            }
        }

        schur_solve_mpi(2, nx, nz, lower, diagonal, upper, known, increment);

        for (int i = 0; i < nx; i++) {
            size_t column = decomp_index(d, i, j, 0);
            size_t line = (size_t)i * (size_t)nz;

            for (int k = 0; k < nz; k++) {
                u[column + (size_t)k * stride_z] += increment[line + (size_t)k];
            }
        }
    }

    free(increment);
    free(known);
    free(upper);
    free(diagonal);
    free(lower);
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
    for (int v_comp = 0; v_comp < 3; v_comp++) {
#if defined(USE_SIMD) && SIMD_AVAILABLE
        /* La versione vettorizzata risolve la linea intera, quindi vale solo
         * finche' Z non e' diviso fra piu' processi. */
        if (decomp->n[2] == decomp->n_global[2]) {
            update_u_simd(decomp, solver_mem_state, rhs, tmp, data, t_step,
                          v_comp, U_SIMD_LINES);
            continue;
        }
#endif
        update_u(decomp, solver_mem_state, data, t_step, v_comp);
    }
    solver_stats->u_sys += time_ns() - start_ns;
}
