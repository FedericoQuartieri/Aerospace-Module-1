#include "pressure.h"
#include "schur.h"
#include "utils.h"

/*
 * Divergenza della velocita', divisa per il passo temporale: e' il termine
 * noto del primo dei tre passi della pressione.
 *
 * Sulle tre facce inferiori del dominio vale zero (la velocita' e' a
 * divergenza nulla). Sono facce globali: un processo che non le tocca calcola
 * la divergenza anche li', leggendo la cella precedente dall'anello di
 * contorno.
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
 * Un passo della cascata della pressione (Lecture 5, p. 7):
 *
 *   (I - d_xx) psi = -div(u)/dt,  poi  (I - d_yy) phi = psi,
 *                                 poi  (I - d_zz) fi  = phi
 *
 * I tre passi hanno la stessa matrice: cambiano solo l'asse, il campo da cui
 * si legge e quello in cui si scrive. Questa funzione ne fa uno, e la si
 * chiama tre volte.
 *
 * Le righe agli estremi sono quelle della condizione di Neumann omogenea
 * (Lecture 5, pp. 15-16), e spettano solo a chi tocca davvero la parete: un
 * processo in mezzo al dominio usa ovunque la riga interna. Sono asimmetriche
 * perche' lo e' la discretizzazione: a sinistra il nodo fantasma dista due
 * mezze celle, a destra una.
 */
static void pressure_direction(const Decomp *restrict d, int axis,
                               const Real *restrict source,
                               Real *restrict target) {
    const int group = (axis == 0) ? 1 : 0;
    const int outer = (axis == 2) ? 1 : 2;
    const int length = d->n[axis];
    const int lines = d->n[group];
    const int last_global = d->n_global[axis] - 1;
    const size_t step = d->stride[axis];
    const Real w = (axis == 0) ? -(Real)DX_INVERSE_SQUARE
                 : (axis == 1) ? -(Real)DY_INVERSE_SQUARE
                               : -(Real)DZ_INVERSE_SQUARE;

    size_t room = (size_t)lines * (size_t)length * sizeof(Real);
    Real *lower = xmalloc(room);
    Real *diagonal = xmalloc(room);
    Real *upper = xmalloc(room);
    Real *known = xmalloc(room);
    Real *answer = xmalloc(room);

    int cell[3];

    for (int b = 0; b < d->n[outer]; b++) {
        cell[outer] = b;

        for (int a = 0; a < lines; a++) {
            cell[group] = a;
            cell[axis] = 0;

            size_t start = decomp_index(d, cell[0], cell[1], cell[2]);
            size_t line = (size_t)a * (size_t)length;

            for (int t = 0; t < length; t++) {
                int along = decomp_global(d, t, axis);
                size_t at = line + (size_t)t;

                if (along == 0) {
                    lower[at] = (Real)0;
                    diagonal[at] = (Real)1 - (Real)2 * w;
                    upper[at] = (Real)2 * w;
                } else if (along == last_global) {
                    lower[at] = w;
                    diagonal[at] = (Real)1 - w;
                    upper[at] = (Real)0;
                } else {
                    lower[at] = w;
                    diagonal[at] = (Real)1 - (Real)2 * w;
                    upper[at] = w;
                }

                known[at] = source[start + (size_t)t * step];
            }
        }

        schur_solve_mpi(axis, lines, length,
                        lower, diagonal, upper, known, answer);

        for (int a = 0; a < lines; a++) {
            cell[group] = a;
            cell[axis] = 0;

            size_t start = decomp_index(d, cell[0], cell[1], cell[2]);
            size_t line = (size_t)a * (size_t)length;

            for (int t = 0; t < length; t++) {
                target[start + (size_t)t * step] = answer[line + (size_t)t];
            }
        }
    }

    free(answer);
    free(known);
    free(upper);
    free(diagonal);
    free(lower);
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
    (void)rhs;
    (void)tmp;

    Real *buffer = pressure_buffer->v;
    Real *star = solver_mem_state->pressure_star.v;

    /* I due campi si scambiano il ruolo a ogni passo, cosi' ne bastano due. */
    uint64_t start_ns = time_ns();
    compute_div(decomp, buffer, &solver_mem_state->u);
    pressure_direction(decomp, 0, buffer, star);      /* psi     */
    solver_stats->psi_sys += time_ns() - start_ns;

    start_ns = time_ns();
    pressure_direction(decomp, 1, star, buffer);      /* phi     */
    solver_stats->phi_low_sys += time_ns() - start_ns;

    start_ns = time_ns();
    pressure_direction(decomp, 2, buffer, star);      /* fi      */
    solver_stats->phi_high_sys += time_ns() - start_ns;

    start_ns = time_ns();
    update_pressure(decomp, solver_mem_state);
    solver_stats->pressure_update += time_ns() - start_ns;
}
