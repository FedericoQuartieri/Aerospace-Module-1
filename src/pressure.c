#include "pressure.h"


static inline void compute_pressure_rhs_x_line(
    Real *restrict rhs,
    const VectorField *restrict u,
    size_t row_offset) {
    const Real *restrict u_x = u->v_x;
    const Real *restrict u_y = u->v_y;
    const Real *restrict u_z = u->v_z;
    const size_t plane_size = (size_t)WIDTH * HEIGHT;
    const Real x_rhs_factor = -(Real)DX_INVERSE / (Real)DT;
    const Real y_rhs_factor = -(Real)DY_INVERSE / (Real)DT;
    const Real z_rhs_factor = -(Real)DZ_INVERSE / (Real)DT;

    rhs[0] = (Real)0;
    for (size_t i = 1; i < WIDTH; i++) {
        const size_t index = row_offset + i;

        rhs[i] =
            (u_x[index] - u_x[index - 1]) * x_rhs_factor +
            (u_y[index] - u_y[index - WIDTH]) * y_rhs_factor +
            (u_z[index] - u_z[index - plane_size]) * z_rhs_factor;
    }
}

void compute_div(Real *restrict u_div,
                 const VectorField *restrict u) {
    const size_t plane_size = (size_t)WIDTH * HEIGHT;

    /*
     * The pressure RHS is zero on the three lower faces (divergence free).
     * Keeping these cases outside the innermost loop leaves the interior
     * kernel branch-free and lets the compiler vectorize its unit-stride
     * accesses.
     */
    for (size_t index = 0; index < plane_size; index++) {
        u_div[index] = (Real)0;
    }

    for (size_t k = 1; k < DEPTH; k++) {
        const size_t plane_offset = k * plane_size;

        for (size_t i = 0; i < WIDTH; i++) {
            u_div[plane_offset + i] = (Real)0;
        }

        for (size_t j = 1; j < HEIGHT; j++) {
            const size_t row_offset = plane_offset + j * (size_t)WIDTH;
            compute_pressure_rhs_x_line(u_div + row_offset, u, row_offset);
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
    const Real inverse_diagonal_left =
        (Real)1 / ((Real)1 - (Real)2 * w);

    tmp[0] = ((Real)2 * w) * inverse_diagonal_left;
    for (size_t index = 1; index + 1 < length; index++) {
        const Real inverse_diagonal =
            (Real)1 /
            (((Real)1 - (Real)2 * w) - w * tmp[index - 1]);
        tmp[index] = w * inverse_diagonal;
    }

    return (Real)1 /
           (((Real)1 - w) - w * tmp[length - 2]);
}

/* pressure_star temporarily stores psi, avoiding another full-size field. */
static void compute_psi(SolverMemState *restrict solver_mem_state,
                        Real *restrict rhs,
                        Real *restrict tmp) {
    const VectorField *restrict u = &solver_mem_state->u;
    Real *restrict psi = solver_mem_state->pressure_star.v;

    const size_t plane_size = (size_t)WIDTH * HEIGHT;
    const Real w = -(Real)DX_INVERSE_SQUARE;
    const Real inverse_w = (Real)1 / w;
    const Real inverse_diagonal_right =
        prepare_pressure_thomas(tmp, WIDTH, w);

    /*
     * For k == 0 or j == 0 the complete RHS line is zero, hence its
     * solution is zero and Thomas can be skipped.
     */
    for (size_t index = 0; index < plane_size; index++) {
        psi[index] = (Real)0;
    }

    for (size_t k = 1; k < DEPTH; k++) {
        const size_t plane_offset = k * plane_size;

        for (size_t i = 0; i < WIDTH; i++) {
            psi[plane_offset + i] = (Real)0;
        }

        for (size_t j = 1; j < HEIGHT; j++) {
            const size_t row_offset =
                plane_offset + j * (size_t)WIDTH;

            /*
             * Build only the current contiguous RHS line.  It is consumed
             * immediately by Thomas and overwritten during elimination.
             */
            compute_pressure_rhs_x_line(rhs, u, row_offset);

            /* Forward elimination. rhs[0] is already zero. */
            for (size_t i = 1; i + 1 < WIDTH; i++) {
                const Real inverse_diagonal = tmp[i] * inverse_w;
                rhs[i] =
                    (rhs[i] - w * rhs[i - 1]) * inverse_diagonal;
            }

            rhs[WIDTH - 1] =
                (rhs[WIDTH - 1] - w * rhs[WIDTH - 2]) *
                inverse_diagonal_right;

            /* Backward substitution, including the i == 0 pressure point. */
            psi[row_offset + WIDTH - 1] = rhs[WIDTH - 1];
            for (size_t i = WIDTH - 1; i-- > 0;) {
                psi[row_offset + i] =
                    rhs[i] - tmp[i] * psi[row_offset + i + 1];
            }
        }
    }
}

static void compute_phi_low(const ScalarField *restrict psi_field,
                            ScalarField *restrict phi_low_field,
                            Real *restrict tmp) {
    const Real *restrict psi = psi_field->v;
    Real *restrict phi_low = phi_low_field->v;
    const size_t plane_size = (size_t)WIDTH * HEIGHT;
    const Real w = -(Real)DY_INVERSE_SQUARE;
    const Real inverse_w = (Real)1 / w;
    const Real inverse_diagonal_left =
        (Real)1 / ((Real)1 - (Real)2 * w);
    const Real inverse_diagonal_right =
        prepare_pressure_thomas(tmp, HEIGHT, w);

    /*
     * Thomas advances along Y, while the inner X loop handles independent
     * systems with contiguous accesses.  This avoids gather/scatter buffers
     * and allows SIMD across X.
     */
    for (size_t k = 0; k < DEPTH; k++) {
        const size_t plane_offset = k * plane_size;

        for (size_t i = 0; i < WIDTH; i++) {
            phi_low[plane_offset + i] =
                psi[plane_offset + i] * inverse_diagonal_left;
        }

        for (size_t j = 1; j + 1 < HEIGHT; j++) {
            const size_t row_offset =
                plane_offset + j * (size_t)WIDTH;
            const size_t previous_row = row_offset - WIDTH;
            const Real inverse_diagonal = tmp[j] * inverse_w;

            for (size_t i = 0; i < WIDTH; i++) {
                phi_low[row_offset + i] =
                    (psi[row_offset + i] -
                     w * phi_low[previous_row + i]) *
                    inverse_diagonal;
            }
        }

        {
            const size_t row_offset =
                plane_offset + (size_t)(HEIGHT - 1) * WIDTH;
            const size_t previous_row = row_offset - WIDTH;

            for (size_t i = 0; i < WIDTH; i++) {
                phi_low[row_offset + i] =
                    (psi[row_offset + i] -
                     w * phi_low[previous_row + i]) *
                    inverse_diagonal_right;
            }
        }

        for (size_t j = HEIGHT - 1; j-- > 0;) {
            const size_t row_offset =
                plane_offset + j * (size_t)WIDTH;
            const size_t next_row = row_offset + WIDTH;

            for (size_t i = 0; i < WIDTH; i++) {
                phi_low[row_offset + i] -=
                    tmp[j] * phi_low[next_row + i];
            }
        }
    }
}

static void compute_phi_high(const ScalarField *restrict phi_low_field,
                             ScalarField *restrict phi_high_field,
                             Real *restrict tmp) {
    const Real *restrict phi_low = phi_low_field->v;
    Real *restrict phi_high = phi_high_field->v;
    const size_t plane_size = (size_t)WIDTH * HEIGHT;
    const Real w = -(Real)DZ_INVERSE_SQUARE;
    const Real inverse_w = (Real)1 / w;
    const Real inverse_diagonal_left =
        (Real)1 / ((Real)1 - (Real)2 * w);
    const Real inverse_diagonal_right =
        prepare_pressure_thomas(tmp, DEPTH, w);

    /*
     * Each Z level is a contiguous XY plane.  Solving all independent
     * columns together turns both Thomas passes into streaming SIMD loops.
     */
    for (size_t index = 0; index < plane_size; index++) {
        phi_high[index] = phi_low[index] * inverse_diagonal_left;
    }

    for (size_t k = 1; k + 1 < DEPTH; k++) {
        const size_t plane_offset = k * plane_size;
        const size_t previous_plane = plane_offset - plane_size;
        const Real inverse_diagonal = tmp[k] * inverse_w;

        for (size_t index = 0; index < plane_size; index++) {
            phi_high[plane_offset + index] =
                (phi_low[plane_offset + index] -
                 w * phi_high[previous_plane + index]) *
                inverse_diagonal;
        }
    }

    {
        const size_t plane_offset = (size_t)(DEPTH - 1) * plane_size;
        const size_t previous_plane = plane_offset - plane_size;

        for (size_t index = 0; index < plane_size; index++) {
            phi_high[plane_offset + index] =
                (phi_low[plane_offset + index] -
                 w * phi_high[previous_plane + index]) *
                inverse_diagonal_right;
        }
    }

    for (size_t k = DEPTH - 1; k-- > 0;) {
        const size_t plane_offset = k * plane_size;
        const size_t next_plane = plane_offset + plane_size;

        for (size_t index = 0; index < plane_size; index++) {
            phi_high[plane_offset + index] -=
                tmp[k] * phi_high[next_plane + index];
        }
    }
}

static void update_pressure(SolverMemState *restrict solver_mem_state) {
    Real *restrict pressure = solver_mem_state->pressure.v;
    Real *restrict phi_high = solver_mem_state->pressure_star.v;

    for (size_t index = 0; index < GRID_CELLS; index++) {
        const Real phi = phi_high[index];
        const Real pressure_new = pressure[index] + phi;

        pressure[index] = pressure_new;
        phi_high[index] = pressure_new + phi;
    }
}

void pressure_step(SolverMemState *solver_mem_state,
                   ScalarField *pressure_buffer,
                   Real *restrict rhs,
                   Real *restrict tmp,
                   SolverStats *solver_stats)
{

    uint64_t start_ns = time_ns();
    compute_psi(solver_mem_state, rhs, tmp);
    solver_stats->psi_sys += time_ns() - start_ns;

    start_ns = time_ns();
    /* RHS: psi in pressure_star; unknown: phi_low in pressure_buffer. */
    compute_phi_low(&solver_mem_state->pressure_star,
                    pressure_buffer, tmp);
    solver_stats->phi_low_sys += time_ns() - start_ns;

    start_ns = time_ns();
    /* Swap roles: phi_low is the RHS, pressure_star receives phi_high. */
    compute_phi_high(pressure_buffer,
                     &solver_mem_state->pressure_star, tmp);
    solver_stats->phi_high_sys += time_ns() - start_ns;

    start_ns = time_ns();
    update_pressure(solver_mem_state);
    solver_stats->pressure_update += time_ns() - start_ns;
}

