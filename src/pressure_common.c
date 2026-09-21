#include "pressure_common.h"
#include "utils.h"
#include "workers.h"

/*
 * The pressure physics that does not depend on how the system is solved.
 *
 * The right-hand side, the matrix of a line and the final update are the same
 * whatever the tridiagonal backend is: only who solves, in the middle,
 * changes. For momentum the shared piece is the row of a *cell*, because gamma
 * follows the permeability point by point; here it is the matrix of a whole
 * *line*, because it depends neither on time nor on which line it is. They are
 * two different forms for a physical reason, not for taste.
 */

/*
 * Divergence of the velocity, divided by the time step: it is the right-hand
 * side of the first of the three pressure steps.
 *
 * On the three lower faces of the domain it is zero (the velocity is
 * divergence-free). They are global faces: a process that does not touch them
 * computes the divergence there too, reading the previous cell from the
 * boundary ring.
 */
void compute_div(const Decomp *restrict d,
                 Real *restrict u_div,
                 const VectorField *restrict u) {
    const Real *restrict u_x = u->v_x;
    const Real *restrict u_y = u->v_y;
    const Real *restrict u_z = u->v_z;
    const size_t stride_y = d->stride[1];
    const size_t stride_z = d->stride[2];
    const Real x_factor = -(Real)DX_INVERSE / (Real)DT;
    const Real y_factor = -(Real)DY_INVERSE / (Real)DT;
    const Real z_factor = -(Real)DZ_INVERSE / (Real)DT;

    WORKERS_PARALLEL_FOR(workers_many())
    for (int k = 0; k < d->n[2]; k++) {
        int gk = decomp_global(d, k, 2);

        for (int j = 0; j < d->n[1]; j++) {
            int gj = decomp_global(d, j, 1);
            size_t row = decomp_index(d, 0, j, k);

            if (gk == 0 || gj == 0) {
                for (int i = 0; i < d->n[0]; i++) {
                    u_div[row + (size_t)i] = (Real)0;
                }
                continue;
            }

            int i0 = d->is_first[0] ? 1 : 0;
            if (d->is_first[0]) {
                u_div[row] = (Real)0;
            }

            for (int i = i0; i < d->n[0]; i++) {
                size_t index = row + (size_t)i;

                u_div[index] =
                    (u_x[index] - u_x[index - 1]) * x_factor +
                    (u_y[index] - u_y[index - stride_y]) * y_factor +
                    (u_z[index] - u_z[index - stride_z]) * z_factor;
            }
        }
    }
}

/*
 * The matrix of a line of the pressure cascade.
 *
 * It is the same for all the lines of the axis and never changes in time: each
 * row depends only on the global position of its point.
 *
 * The rows at the ends are those of the homogeneous Neumann condition (Lecture
 * 5, pp. 15-16), and belong only to whoever really touches the wall: a process
 * in the middle of the domain uses the interior row everywhere. They are
 * asymmetric because the discretisation is: on the left the ghost node is two
 * half cells away, on the right one.
 */
void pressure_matrix(const Decomp *restrict d, int axis,
                            Real *restrict a,
                            Real *restrict b,
                            Real *restrict c) {
    const int length = d->n[axis];
    const int last_global = d->n_global[axis] - 1;
    const Real w = (axis == 0) ? -(Real)DX_INVERSE_SQUARE
                 : (axis == 1) ? -(Real)DY_INVERSE_SQUARE
                               : -(Real)DZ_INVERSE_SQUARE;

    for (int t = 0; t < length; t++) {
        int along = decomp_global(d, t, axis);

        if (along == 0) {
            a[t] = (Real)0;
            b[t] = (Real)1 - (Real)2 * w;
            c[t] = (Real)2 * w;
        } else if (along == last_global) {
            a[t] = w;
            b[t] = (Real)1 - w;
            c[t] = (Real)0;
        } else {
            a[t] = w;
            b[t] = (Real)1 - (Real)2 * w;
            c[t] = w;
        }
    }
}

void update_pressure(const Decomp *restrict d,
                            SolverMemState *restrict solver_mem_state) {
    Real *restrict pressure = solver_mem_state->pressure.v;
    Real *restrict phi_high = solver_mem_state->pressure_star.v;

    WORKERS_PARALLEL_FOR(workers_many())
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
