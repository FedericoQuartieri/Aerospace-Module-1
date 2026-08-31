#include "momentum.h"
#include "workers.h"


#if defined(USE_SIMD) && SIMD_AVAILABLE

/* Compute the Thomas coefficient w from one vector of permeability values. */
static inline SimdReal weight_from_k_simd(SimdReal k,
                                          SimdReal numerator,
                                          SimdReal negative_inverse_square,
                                          SimdReal one,
                                          SimdReal two) {
    SimdReal beta =
        simd_add(one, simd_div(numerator, simd_mul(two, k)));
    SimdReal gamma =
        simd_div(numerator, simd_mul(two, beta));

    return simd_mul(gamma, negative_inverse_square);
}

/* Solve the x lines left after the last complete SIMD vector of Dyy. */
static void update_zeta_scalar_tail(const Decomp *restrict d,
                                    const Real *restrict k_porosity,
                                    const Real *restrict eta,
                                    Real *restrict zeta,
                                    Real *restrict rhs,
                                    Real *restrict tmp,
                                    Data *data,
                                    int t_step,
                                    int v_comp,
                                    int k,
                                    int first_i,
                                    bool same_direction) {
    const int ny = d->n[1];
    const size_t stride_y = d->stride[1];
    const int gk = decomp_global(d, k, 2);

    for (int i = first_i; i < d->n[0]; i++) {
        size_t off = decomp_index(d, i, 0, k);
        const int gi = decomp_global(d, i, 0);

        tmp[0] = 0.0;
        rhs[0] = bc_left(data->bc_velocity,
                         gi, decomp_global(d, 0, 1), gk, t_step, v_comp);

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

        size_t j_arr = off + (size_t)(ny-1) * stride_y;
        Real k_i = k_porosity[j_arr];
        Real w_i = -gamma_from_k(k_i) * DY_INVERSE_SQUARE;
        Real right_value =
            bc_right(data->bc_velocity,
                     gi, decomp_global(d, ny-1, 1), gk, t_step, v_comp);

        rhs[ny-1] = eta[j_arr] - zeta[j_arr];
        rhs[ny-1] -= 2.0 * w_i * right_value;

        Real norm_coeff =
            1.0 / ((1.0 - 3.0 * w_i) - w_i * tmp[ny - 2]);
        rhs[ny-1] =
            (rhs[ny-1] - w_i * rhs[ny-2]) * norm_coeff;

        Real up1 = same_direction ? right_value : rhs[ny-1];

        zeta[j_arr] += up1;
        for (int j = ny-2; j >= 0; j--) {
            j_arr = off + (size_t)j * stride_y;
            Real up2 = rhs[j] - tmp[j] * up1;

            zeta[j_arr] += up2;
            up1 = up2;
        }
    }
}

/* Solve the x lines left after the last complete SIMD vector of Dzz. */
static void update_u_scalar_tail(const Decomp *restrict d,
                                 const Real *restrict k_porosity,
                                 const Real *restrict zeta,
                                 Real *restrict u,
                                 Real *restrict rhs,
                                 Real *restrict tmp,
                                 Data *data,
                                 int t_step,
                                 int v_comp,
                                 int j,
                                 int first_i,
                                 bool same_direction) {
    const int nz = d->n[2];
    const size_t stride_z = d->stride[2];
    const int gj = decomp_global(d, j, 1);

    for (int i = first_i; i < d->n[0]; i++) {
        size_t off = decomp_index(d, i, j, 0);
        const int gi = decomp_global(d, i, 0);

        tmp[0] = 0.0;
        rhs[0] = bc_left(data->bc_velocity,
                         gi, gj, decomp_global(d, 0, 2), t_step, v_comp);

        for (int k = 1; k < nz-1; k++) {
            size_t k_arr = off + (size_t)k * stride_z;
            Real k_i = k_porosity[k_arr];
            Real w_i = -gamma_from_k(k_i) * DZ_INVERSE_SQUARE;
            Real norm_coeff =
                1.0 / ((1.0 - 2.0 * w_i) - w_i * tmp[k - 1]);

            tmp[k] = w_i * norm_coeff;
            rhs[k] = zeta[k_arr] - u[k_arr];
            rhs[k] = (rhs[k] - w_i * rhs[k - 1]) * norm_coeff;
        }

        size_t k_arr = off + (size_t)(nz-1) * stride_z;
        Real k_i = k_porosity[k_arr];
        Real w_i = -gamma_from_k(k_i) * DZ_INVERSE_SQUARE;
        Real right_value =
            bc_right(data->bc_velocity,
                     gi, gj, decomp_global(d, nz-1, 2), t_step, v_comp);

        rhs[nz-1] = zeta[k_arr] - u[k_arr];
        rhs[nz-1] -= 2.0 * w_i * right_value;

        Real norm_coeff =
            1.0 / ((1.0 - 3.0 * w_i) - w_i * tmp[nz - 2]);
        rhs[nz-1] =
            (rhs[nz-1] - w_i * rhs[nz-2]) * norm_coeff;

        Real up1 = same_direction ? right_value : rhs[nz-1];

        u[k_arr] += up1;
        for (int k = nz-2; k >= 0; k--) {
            k_arr = off + (size_t)k * stride_z;
            Real up2 = rhs[k] - tmp[k] * up1;

            u[k_arr] += up2;
            up1 = up2;
        }
    }
}

/* Solve Dyy along y, assigning adjacent independent x lines to SIMD lanes. */
void update_zeta_simd(const Decomp *d,
                      SolverMemState *solver_mem_state,
                      Real *restrict rhs_pool,
                      Real *restrict tmp_pool,
                      Data *data, int t_step, int v_comp,
                      int simd_lines) {
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
    int max_vectors = simd_lines / SIMD_LANES;
    const int nx = d->n[0];
    const int ny = d->n[1];
    const size_t stride_y = d->stride[1];
    SimdReal one = simd_set1((Real)1.0);
    SimdReal two = simd_set1((Real)2.0);
    SimdReal three = simd_set1((Real)3.0);
    SimdReal numerator = simd_set1((Real)(DT * NU));
    SimdReal negative_inverse_square =
        simd_set1((Real)-DY_INVERSE_SQUARE);

    /* I piani k sono indipendenti; gli scratch no, quindi ogni thread prende
     * la propria fetta di quelli che solver_solve ha allocato in fila. */
    const size_t slice = momentum_scratch_slice(d);
    const int slots = workers_slots(true, d->n[2]);

    WORKERS_PARALLEL_FOR(slots > 1)
    for (int k = 0; k < d->n[2]; k++) {
        Real *restrict rhs = rhs_pool + (size_t)workers_slot(slots) * slice;
        Real *restrict tmp = tmp_pool + (size_t)workers_slot(slots) * slice;
        int i = 0;
        const int gk = decomp_global(d, k, 2);

        while (max_vectors > 0 && i + SIMD_LANES <= nx) {
            int remaining_vectors = (nx - i) / SIMD_LANES;
            int active_vectors =
                remaining_vectors < max_vectors
                    ? remaining_vectors : max_vectors;
            int active_lines = active_vectors * SIMD_LANES;
            size_t block_offset = decomp_index(d, i, 0, k);

            for (int lane = 0; lane < active_lines; lane++) {
                tmp[lane] = 0.0;
                rhs[lane] =
                    bc_left(data->bc_velocity,
                            decomp_global(d, i + lane, 0),
                            decomp_global(d, 0, 1), gk,
                            t_step, v_comp);
            }

            for (int j = 1; j < ny-1; j++) {
                size_t level_offset =
                    block_offset + (size_t)j * stride_y;

                for (int vector = 0; vector < active_vectors; vector++) {
                    int lane = vector * SIMD_LANES;
                    size_t field_index = level_offset + (size_t)lane;
                    size_t scratch_index =
                        (size_t)j * active_lines + (size_t)lane;
                    size_t previous_scratch =
                        (size_t)(j - 1) * active_lines + (size_t)lane;
                    SimdReal k_i = simd_loadu(&k_porosity[field_index]);
                    SimdReal w_i =
                        weight_from_k_simd(k_i, numerator,
                                           negative_inverse_square,
                                           one, two);
                    SimdReal tmp_previous =
                        simd_loadu(&tmp[previous_scratch]);
                    SimdReal denominator =
                        simd_sub(simd_sub(one, simd_mul(two, w_i)),
                                 simd_mul(w_i, tmp_previous));
                    SimdReal norm_coeff =
                        simd_div(one, denominator);

                    simd_storeu(&tmp[scratch_index],
                                simd_mul(w_i, norm_coeff));

                    SimdReal raw_rhs =
                        simd_sub(simd_loadu(&eta[field_index]),
                                 simd_loadu(&zeta[field_index]));
                    SimdReal rhs_previous =
                        simd_loadu(&rhs[previous_scratch]);
                    SimdReal rhs_current =
                        simd_mul(simd_sub(raw_rhs,
                                          simd_mul(w_i, rhs_previous)),
                                 norm_coeff);

                    simd_storeu(&rhs[scratch_index], rhs_current);
                }
            }

            /* Thomas does not use tmp at the last level: keep the right
             * boundary block there instead of allocating a third buffer. */
            Real *right_boundary =
                &tmp[(size_t)(ny - 1) * active_lines];
            for (int lane = 0; lane < active_lines; lane++) {
                right_boundary[lane] =
                    bc_right(data->bc_velocity,
                             decomp_global(d, i + lane, 0),
                             decomp_global(d, ny-1, 1), gk,
                             t_step, v_comp);
            }

            size_t last_offset =
                block_offset + (size_t)(ny - 1) * stride_y;
            for (int vector = 0; vector < active_vectors; vector++) {
                int lane = vector * SIMD_LANES;
                size_t field_index = last_offset + (size_t)lane;
                size_t last_scratch =
                    (size_t)(ny - 1) * active_lines + (size_t)lane;
                SimdReal up = simd_loadu(&right_boundary[lane]);

                if (!same_direction) {
                    size_t previous_scratch =
                        (size_t)(ny - 2) * active_lines + (size_t)lane;
                    SimdReal k_i = simd_loadu(&k_porosity[field_index]);
                    SimdReal w_i =
                        weight_from_k_simd(k_i, numerator,
                                           negative_inverse_square,
                                           one, two);
                    SimdReal rhs_last =
                        simd_sub(simd_loadu(&eta[field_index]),
                                 simd_loadu(&zeta[field_index]));

                    rhs_last =
                        simd_sub(rhs_last,
                                 simd_mul(simd_mul(two, w_i), up));

                    SimdReal denominator =
                        simd_sub(simd_sub(one, simd_mul(three, w_i)),
                                 simd_mul(w_i,
                                          simd_loadu(&tmp[previous_scratch])));
                    SimdReal norm_coeff =
                        simd_div(one, denominator);

                    up = simd_mul(
                        simd_sub(rhs_last,
                                 simd_mul(w_i,
                                          simd_loadu(&rhs[previous_scratch]))),
                        norm_coeff);
                }

                /* The backward pass overwrites the eliminated rhs with the
                 * solution increment, so no update buffer is necessary. */
                simd_storeu(&rhs[last_scratch], up);
                simd_storeu(&zeta[field_index],
                            simd_add(simd_loadu(&zeta[field_index]), up));
            }

            for (int j = ny-2; j >= 0; j--) {
                size_t level_offset =
                    block_offset + (size_t)j * stride_y;

                for (int vector = 0; vector < active_vectors; vector++) {
                    int lane = vector * SIMD_LANES;
                    size_t field_index = level_offset + (size_t)lane;
                    size_t scratch_index =
                        (size_t)j * active_lines + (size_t)lane;
                    size_t next_scratch =
                        (size_t)(j + 1) * active_lines + (size_t)lane;
                    SimdReal up =
                        simd_sub(simd_loadu(&rhs[scratch_index]),
                                 simd_mul(simd_loadu(&tmp[scratch_index]),
                                          simd_loadu(&rhs[next_scratch])));

                    simd_storeu(&rhs[scratch_index], up);
                    simd_storeu(&zeta[field_index],
                                simd_add(simd_loadu(&zeta[field_index]), up));
                }
            }

            i += active_lines;
        }

        update_zeta_scalar_tail(d, k_porosity, eta, zeta, rhs, tmp,
                                data, t_step, v_comp, k, i,
                                same_direction);
    }
}

/* Solve Dzz along z, assigning adjacent independent x lines to SIMD lanes. */
void update_u_simd(const Decomp *d,
                   SolverMemState *solver_mem_state,
                   Real *restrict rhs_pool,
                   Real *restrict tmp_pool,
                   Data *data, int t_step, int v_comp,
                   int simd_lines) {
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
    int max_vectors = simd_lines / SIMD_LANES;
    const int nx = d->n[0];
    const int nz = d->n[2];
    const size_t stride_z = d->stride[2];
    SimdReal one = simd_set1((Real)1.0);
    SimdReal two = simd_set1((Real)2.0);
    SimdReal three = simd_set1((Real)3.0);
    SimdReal numerator = simd_set1((Real)(DT * NU));
    SimdReal negative_inverse_square =
        simd_set1((Real)-DZ_INVERSE_SQUARE);

    const size_t slice = momentum_scratch_slice(d);
    const int slots = workers_slots(true, d->n[1]);

    WORKERS_PARALLEL_FOR(slots > 1)
    for (int j = 0; j < d->n[1]; j++) {
        Real *restrict rhs = rhs_pool + (size_t)workers_slot(slots) * slice;
        Real *restrict tmp = tmp_pool + (size_t)workers_slot(slots) * slice;
        int i = 0;
        const int gj = decomp_global(d, j, 1);

        while (max_vectors > 0 && i + SIMD_LANES <= nx) {
            int remaining_vectors = (nx - i) / SIMD_LANES;
            int active_vectors =
                remaining_vectors < max_vectors
                    ? remaining_vectors : max_vectors;
            int active_lines = active_vectors * SIMD_LANES;
            size_t block_offset = decomp_index(d, i, j, 0);

            for (int lane = 0; lane < active_lines; lane++) {
                tmp[lane] = 0.0;
                rhs[lane] =
                    bc_left(data->bc_velocity,
                            decomp_global(d, i + lane, 0), gj,
                            decomp_global(d, 0, 2),
                            t_step, v_comp);
            }

            for (int k = 1; k < nz-1; k++) {
                size_t level_offset =
                    block_offset + (size_t)k * stride_z;

                for (int vector = 0; vector < active_vectors; vector++) {
                    int lane = vector * SIMD_LANES;
                    size_t field_index = level_offset + (size_t)lane;
                    size_t scratch_index =
                        (size_t)k * active_lines + (size_t)lane;
                    size_t previous_scratch =
                        (size_t)(k - 1) * active_lines + (size_t)lane;
                    SimdReal k_i = simd_loadu(&k_porosity[field_index]);
                    SimdReal w_i =
                        weight_from_k_simd(k_i, numerator,
                                           negative_inverse_square,
                                           one, two);
                    SimdReal tmp_previous =
                        simd_loadu(&tmp[previous_scratch]);
                    SimdReal denominator =
                        simd_sub(simd_sub(one, simd_mul(two, w_i)),
                                 simd_mul(w_i, tmp_previous));
                    SimdReal norm_coeff =
                        simd_div(one, denominator);

                    simd_storeu(&tmp[scratch_index],
                                simd_mul(w_i, norm_coeff));

                    SimdReal raw_rhs =
                        simd_sub(simd_loadu(&zeta[field_index]),
                                 simd_loadu(&u[field_index]));
                    SimdReal rhs_previous =
                        simd_loadu(&rhs[previous_scratch]);
                    SimdReal rhs_current =
                        simd_mul(simd_sub(raw_rhs,
                                          simd_mul(w_i, rhs_previous)),
                                 norm_coeff);

                    simd_storeu(&rhs[scratch_index], rhs_current);
                }
            }

            Real *right_boundary =
                &tmp[(size_t)(nz - 1) * active_lines];
            for (int lane = 0; lane < active_lines; lane++) {
                right_boundary[lane] =
                    bc_right(data->bc_velocity,
                             decomp_global(d, i + lane, 0), gj,
                             decomp_global(d, nz-1, 2),
                             t_step, v_comp);
            }

            size_t last_offset =
                block_offset + (size_t)(nz - 1) * stride_z;
            for (int vector = 0; vector < active_vectors; vector++) {
                int lane = vector * SIMD_LANES;
                size_t field_index = last_offset + (size_t)lane;
                size_t last_scratch =
                    (size_t)(nz - 1) * active_lines + (size_t)lane;
                SimdReal up = simd_loadu(&right_boundary[lane]);

                if (!same_direction) {
                    size_t previous_scratch =
                        (size_t)(nz - 2) * active_lines + (size_t)lane;
                    SimdReal k_i = simd_loadu(&k_porosity[field_index]);
                    SimdReal w_i =
                        weight_from_k_simd(k_i, numerator,
                                           negative_inverse_square,
                                           one, two);
                    SimdReal rhs_last =
                        simd_sub(simd_loadu(&zeta[field_index]),
                                 simd_loadu(&u[field_index]));

                    rhs_last =
                        simd_sub(rhs_last,
                                 simd_mul(simd_mul(two, w_i), up));

                    SimdReal denominator =
                        simd_sub(simd_sub(one, simd_mul(three, w_i)),
                                 simd_mul(w_i,
                                          simd_loadu(&tmp[previous_scratch])));
                    SimdReal norm_coeff =
                        simd_div(one, denominator);

                    up = simd_mul(
                        simd_sub(rhs_last,
                                 simd_mul(w_i,
                                          simd_loadu(&rhs[previous_scratch]))),
                        norm_coeff);
                }

                simd_storeu(&rhs[last_scratch], up);
                simd_storeu(&u[field_index],
                            simd_add(simd_loadu(&u[field_index]), up));
            }

            for (int k = nz-2; k >= 0; k--) {
                size_t level_offset =
                    block_offset + (size_t)k * stride_z;

                for (int vector = 0; vector < active_vectors; vector++) {
                    int lane = vector * SIMD_LANES;
                    size_t field_index = level_offset + (size_t)lane;
                    size_t scratch_index =
                        (size_t)k * active_lines + (size_t)lane;
                    size_t next_scratch =
                        (size_t)(k + 1) * active_lines + (size_t)lane;
                    SimdReal up =
                        simd_sub(simd_loadu(&rhs[scratch_index]),
                                 simd_mul(simd_loadu(&tmp[scratch_index]),
                                          simd_loadu(&rhs[next_scratch])));

                    simd_storeu(&rhs[scratch_index], up);
                    simd_storeu(&u[field_index],
                                simd_add(simd_loadu(&u[field_index]), up));
                }
            }

            i += active_lines;
        }

        update_u_scalar_tail(d, k_porosity, zeta, u, rhs, tmp,
                             data, t_step, v_comp, j, i,
                             same_direction);
    }
}

#endif
