#include "pressure.h"


static inline void compute_pressure_rhs_x_line(
    const Decomp *restrict d,
    Real *restrict rhs,
    const VectorField *restrict u,
    size_t row_offset) {
    const Real *restrict u_x = u->v_x;
    const Real *restrict u_y = u->v_y;
    const Real *restrict u_z = u->v_z;
    size_t stride_y = d->stride[1];
    size_t stride_z = d->stride[2];
    int nx = d->n[0];
    Real x_rhs_factor = -(Real)DX_INVERSE / (Real)DT;
    Real y_rhs_factor = -(Real)DY_INVERSE / (Real)DT;
    Real z_rhs_factor = -(Real)DZ_INVERSE / (Real)DT;

    rhs[0] = (Real)0;
    for (int i = 1; i < nx; i++) {
        size_t index = row_offset + (size_t)i;

        rhs[i] =
            (u_x[index] - u_x[index - 1]) * x_rhs_factor +
            (u_y[index] - u_y[index - stride_y]) * y_rhs_factor +
            (u_z[index] - u_z[index - stride_z]) * z_rhs_factor;
    }
}

void compute_div(const Decomp *restrict d,
                 Real *restrict u_div,
                 const VectorField *restrict u) {
    /*
     * The pressure RHS is zero on the three lower faces (divergence free).
     * Keeping these cases outside the innermost loop leaves the interior
     * kernel branch-free and lets the compiler vectorize its unit-stride
     * accesses.  The faces belong to the global domain, so a block that does
     * not touch them starts its loops one cell earlier.
     */
    int k0 = d->is_first[2] ? 1 : 0;
    int j0 = d->is_first[1] ? 1 : 0;

    if (d->is_first[2]) {
        for (int j = 0; j < d->n[1]; j++) {
            size_t row = decomp_index(d, 0, j, 0);
            for (int i = 0; i < d->n[0]; i++) {
                u_div[row + (size_t)i] = (Real)0;
            }
        }
    }

    for (int k = k0; k < d->n[2]; k++) {
        if (d->is_first[1]) {
            size_t row = decomp_index(d, 0, 0, k);
            for (int i = 0; i < d->n[0]; i++) {
                u_div[row + (size_t)i] = (Real)0;
            }
        }

        for (int j = j0; j < d->n[1]; j++) {
            size_t row_offset = decomp_index(d, 0, j, k);
            compute_pressure_rhs_x_line(d, u_div + row_offset, u, row_offset);
        }
    }
}

/*
 * The pressure matrices have constant coefficients.  Build the normalized
 * superdiagonal once per directional solve instead of repeating divisions
 * for every independent line.
 */
static Real prepare_pressure_thomas(Real *restrict tmp,
                                    size_t length,
                                    Real w) {
    Real inverse_diagonal_left =
        (Real)1 / ((Real)1 - (Real)2 * w);

    tmp[0] = ((Real)2 * w) * inverse_diagonal_left;
    for (size_t index = 1; index + 1 < length; index++) {
        Real inverse_diagonal =
            (Real)1 /
            (((Real)1 - (Real)2 * w) - w * tmp[index - 1]);
        tmp[index] = w * inverse_diagonal;
    }

    return (Real)1 /
           (((Real)1 - w) - w * tmp[length - 2]);
}

/* pressure_star temporarily stores psi, avoiding another full-size field. */
static void compute_psi(const Decomp *restrict d,
                        SolverMemState *restrict solver_mem_state,
                        Real *restrict rhs,
                        Real *restrict tmp) {
    const VectorField *restrict u = &solver_mem_state->u;
    Real *restrict psi = solver_mem_state->pressure_star.v;

    int nx = d->n[0];
    Real w = -(Real)DX_INVERSE_SQUARE;
    Real inverse_w = (Real)1 / w;
    Real inverse_diagonal_right =
        prepare_pressure_thomas(tmp, (size_t)nx, w);

    /*
     * For k == 0 or j == 0 the complete RHS line is zero, hence its
     * solution is zero and Thomas can be skipped.
     */
    int k0 = d->is_first[2] ? 1 : 0;
    int j0 = d->is_first[1] ? 1 : 0;

    if (d->is_first[2]) {
        for (int j = 0; j < d->n[1]; j++) {
            size_t row = decomp_index(d, 0, j, 0);
            for (int i = 0; i < nx; i++) {
                psi[row + (size_t)i] = (Real)0;
            }
        }
    }

    for (int k = k0; k < d->n[2]; k++) {
        if (d->is_first[1]) {
            size_t row = decomp_index(d, 0, 0, k);
            for (int i = 0; i < nx; i++) {
                psi[row + (size_t)i] = (Real)0;
            }
        }

        for (int j = j0; j < d->n[1]; j++) {
            size_t row_offset = decomp_index(d, 0, j, k);

            /*
             * Build only the current contiguous RHS line.  It is consumed
             * immediately by Thomas and overwritten during elimination.
             */
            compute_pressure_rhs_x_line(d, rhs, u, row_offset);

            /* Forward elimination. rhs[0] is already zero. */
            for (int i = 1; i + 1 < nx; i++) {
                Real inverse_diagonal = tmp[i] * inverse_w;
                rhs[i] =
                    (rhs[i] - w * rhs[i - 1]) * inverse_diagonal;
            }

            rhs[nx - 1] =
                (rhs[nx - 1] - w * rhs[nx - 2]) *
                inverse_diagonal_right;

            /* Backward substitution, including the i == 0 pressure point. */
            psi[row_offset + (size_t)(nx - 1)] = rhs[nx - 1];
            for (int i = nx - 1; i-- > 0;) {
                psi[row_offset + (size_t)i] =
                    rhs[i] - tmp[i] * psi[row_offset + (size_t)i + 1];
            }
        }
    }
}

static void compute_phi_low(const Decomp *restrict d,
                            const ScalarField *restrict psi_field,
                            ScalarField *restrict phi_low_field,
                            Real *restrict tmp) {
    const Real *restrict psi = psi_field->v;
    Real *restrict phi_low = phi_low_field->v;
    int nx = d->n[0];
    int ny = d->n[1];
    size_t stride_y = d->stride[1];
    Real w = -(Real)DY_INVERSE_SQUARE;
    Real inverse_w = (Real)1 / w;
    Real inverse_diagonal_left =
        (Real)1 / ((Real)1 - (Real)2 * w);
    Real inverse_diagonal_right =
        prepare_pressure_thomas(tmp, (size_t)ny, w);

    /*
     * Thomas advances along Y, while the inner X loop handles independent
     * systems with contiguous accesses.  This avoids gather/scatter buffers
     * and allows SIMD across X.
     */
    for (int k = 0; k < d->n[2]; k++) {
        size_t first_row = decomp_index(d, 0, 0, k);

        for (int i = 0; i < nx; i++) {
            phi_low[first_row + (size_t)i] =
                psi[first_row + (size_t)i] * inverse_diagonal_left;
        }

        for (int j = 1; j + 1 < ny; j++) {
            size_t row_offset = first_row + (size_t)j * stride_y;
            size_t previous_row = row_offset - stride_y;
            Real inverse_diagonal = tmp[j] * inverse_w;

            for (int i = 0; i < nx; i++) {
                phi_low[row_offset + (size_t)i] =
                    (psi[row_offset + (size_t)i] -
                     w * phi_low[previous_row + (size_t)i]) *
                    inverse_diagonal;
            }
        }

        {
            size_t row_offset = first_row + (size_t)(ny - 1) * stride_y;
            size_t previous_row = row_offset - stride_y;

            for (int i = 0; i < nx; i++) {
                phi_low[row_offset + (size_t)i] =
                    (psi[row_offset + (size_t)i] -
                     w * phi_low[previous_row + (size_t)i]) *
                    inverse_diagonal_right;
            }
        }

        for (int j = ny - 1; j-- > 0;) {
            size_t row_offset = first_row + (size_t)j * stride_y;
            size_t next_row = row_offset + stride_y;

            for (int i = 0; i < nx; i++) {
                phi_low[row_offset + (size_t)i] -=
                    tmp[j] * phi_low[next_row + (size_t)i];
            }
        }
    }
}

static void compute_phi_high(const Decomp *restrict d,
                             const ScalarField *restrict phi_low_field,
                             ScalarField *restrict phi_high_field,
                             Real *restrict tmp) {
    const Real *restrict phi_low = phi_low_field->v;
    Real *restrict phi_high = phi_high_field->v;
    int nx = d->n[0];
    int ny = d->n[1];
    int nz = d->n[2];
    size_t stride_z = d->stride[2];
    Real w = -(Real)DZ_INVERSE_SQUARE;
    Real inverse_w = (Real)1 / w;
    Real inverse_diagonal_left =
        (Real)1 / ((Real)1 - (Real)2 * w);
    Real inverse_diagonal_right =
        prepare_pressure_thomas(tmp, (size_t)nz, w);

    /*
     * Thomas advances along Z; the two inner loops sweep the XY plane, whose
     * rows are contiguous, so both passes stay streaming SIMD loops.
     */
    for (int j = 0; j < ny; j++) {
        size_t row = decomp_index(d, 0, j, 0);
        for (int i = 0; i < nx; i++) {
            phi_high[row + (size_t)i] =
                phi_low[row + (size_t)i] * inverse_diagonal_left;
        }
    }

    for (int k = 1; k + 1 < nz; k++) {
        Real inverse_diagonal = tmp[k] * inverse_w;

        for (int j = 0; j < ny; j++) {
            size_t row = decomp_index(d, 0, j, k);
            size_t previous_row = row - stride_z;

            for (int i = 0; i < nx; i++) {
                phi_high[row + (size_t)i] =
                    (phi_low[row + (size_t)i] -
                     w * phi_high[previous_row + (size_t)i]) *
                    inverse_diagonal;
            }
        }
    }

    for (int j = 0; j < ny; j++) {
        size_t row = decomp_index(d, 0, j, nz - 1);
        size_t previous_row = row - stride_z;

        for (int i = 0; i < nx; i++) {
            phi_high[row + (size_t)i] =
                (phi_low[row + (size_t)i] -
                 w * phi_high[previous_row + (size_t)i]) *
                inverse_diagonal_right;
        }
    }

    for (int k = nz - 1; k-- > 0;) {
        for (int j = 0; j < ny; j++) {
            size_t row = decomp_index(d, 0, j, k);
            size_t next_row = row + stride_z;

            for (int i = 0; i < nx; i++) {
                phi_high[row + (size_t)i] -=
                    tmp[k] * phi_high[next_row + (size_t)i];
            }
        }
    }
}

static void update_pressure(const Decomp *restrict d,
                            SolverMemState *restrict solver_mem_state) {
    Real *restrict pressure = solver_mem_state->pressure.v;
    Real *restrict phi_high = solver_mem_state->pressure_star.v;

    for (int k = 0; k < d->n[2]; k++) {
        for (int j = 0; j < d->n[1]; j++) {
            size_t row = decomp_index(d, 0, j, k);

            for (int i = 0; i < d->n[0]; i++) {
                size_t index = row + (size_t)i;
                Real phi = phi_high[index];
                Real pressure_new = pressure[index] + phi;

                pressure[index] = pressure_new;
                phi_high[index] = pressure_new + phi;
            }
        }
    }
}

void pressure_step(const Decomp *decomp,
                   SolverMemState *solver_mem_state,
                   ScalarField *pressure_buffer,
                   Real *restrict rhs,
                   Real *restrict tmp,
                   SolverStats *solver_stats)
{

    uint64_t start_ns = time_ns();
    compute_psi(decomp, solver_mem_state, rhs, tmp);
    solver_stats->psi_sys += time_ns() - start_ns;

    start_ns = time_ns();
    /* RHS: psi in pressure_star; unknown: phi_low in pressure_buffer. */
    compute_phi_low(decomp, &solver_mem_state->pressure_star,
                    pressure_buffer, tmp);
    solver_stats->phi_low_sys += time_ns() - start_ns;

    start_ns = time_ns();
    /* Swap roles: phi_low is the RHS, pressure_star receives phi_high. */
    compute_phi_high(decomp, pressure_buffer,
                     &solver_mem_state->pressure_star, tmp);
    solver_stats->phi_high_sys += time_ns() - start_ns;

    start_ns = time_ns();
    update_pressure(decomp, solver_mem_state);
    solver_stats->pressure_update += time_ns() - start_ns;
}
